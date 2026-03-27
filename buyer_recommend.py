"""
Direction 2: Buyer Recommendation.

Recommend model-level options, not individual historical sale records.
The ranking combines pricing-model value, mileage/year fit, and confidence
based on the number of comparable examples in the dataset.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

try:
    from .predict import _brand_tier, _normalize_brand, get_price_range
except ImportError:
    from predict import _brand_tier, _normalize_brand, get_price_range


_BAD_TEXT_PATTERN = re.compile(
    r"auction|no reserve|salvage|rebuilt|parts only|parts car|rebuilt title|"
    r"bill of sale|not running|project car",
    re.IGNORECASE,
)

_BAD_MODEL_PATTERN = re.compile(r"^other\b|^nan$", re.IGNORECASE)

_BODY_TYPE_MAP = {
    "suv": "SUV",
    "sport utility": "SUV",
    "sport utility vehicle": "SUV",
    "sedan": "Sedan",
    "4dr car": "Sedan",
    "4 door": "Sedan",
    "coupe": "Coupe",
    "2dr car": "Coupe",
    "convertible": "Convertible",
    "hatchback": "Hatchback",
    "wagon": "Wagon",
    "station wagon": "Wagon",
    "minivan": "Minivan",
    "minivan/van": "Minivan",
    "mini-van, passenger": "Minivan",
    "pickup truck": "Pickup",
    "pickup": "Pickup",
    "truck": "Pickup",
    "standard cab pickup": "Pickup",
    "crew cab pickup": "Pickup",
    "extended cab pickup": "Pickup",
    "extended crew cab pickup": "Pickup",
    "regular cab pickup": "Pickup",
}

_DRIVE_TYPE_MAP = {
    "awd": "AWD",
    "all wheel drive": "AWD",
    "all-wheel-drive": "AWD",
    "4matic®": "AWD",
    "fwd": "FWD",
    "front wheel drive": "FWD",
    "front-wheel drive": "FWD",
    "rwd": "RWD",
    "rear": "RWD",
    "rear wheel drive": "RWD",
    "rear wheel": "RWD",
    "rear-wheel drive": "RWD",
    "4wd": "4WD",
    "4x4": "4WD",
    "four wheel drive": "4WD",
}


def _normalize_body_type(raw):
    if pd.isna(raw):
        return "Other"
    text = str(raw).strip().lower()
    text = re.sub(r"\s+", " ", text)
    for key, value in _BODY_TYPE_MAP.items():
        if text == key:
            return value
    if "pickup" in text or "truck" in text:
        return "Pickup"
    if "suv" in text or "sport utility" in text:
        return "SUV"
    if "sedan" in text:
        return "Sedan"
    if "coupe" in text:
        return "Coupe"
    if "convertible" in text:
        return "Convertible"
    if "wagon" in text:
        return "Wagon"
    if "hatch" in text:
        return "Hatchback"
    if "van" in text:
        return "Minivan"
    return str(raw).strip() if str(raw).strip() and str(raw).strip() != "nan" else "Other"


def _normalize_drive_type(raw):
    if pd.isna(raw):
        return "Other"
    text = str(raw).strip().lower()
    text = re.sub(r"\s+", " ", text)
    for key, value in _DRIVE_TYPE_MAP.items():
        if text == key:
            return value
    if "all wheel" in text or "awd" in text or "4matic" in text:
        return "AWD"
    if "four wheel" in text or "4wd" in text or "4x4" in text:
        return "4WD"
    if "front wheel" in text or text == "fwd":
        return "FWD"
    if "rear wheel" in text or text == "rwd" or text == "rear":
        return "RWD"
    return "Other"


def _has_bad_text(row):
    text_parts = [
        row.get("Trim", ""),
        row.get("Engine", ""),
        row.get("Model", ""),
        row.get("BodyType", ""),
        row.get("DriveType", ""),
    ]
    joined = " ".join("" if pd.isna(part) else str(part) for part in text_parts)
    return bool(_BAD_TEXT_PATTERN.search(joined))


def _confidence_label(sample_count):
    if sample_count >= 100:
        return "High"
    if sample_count >= 30:
        return "Medium"
    return "Low"


def _confidence_score(sample_count):
    if sample_count >= 100:
        return 1.0
    if sample_count >= 30:
        return 0.7
    if sample_count >= 15:
        return 0.45
    return 0.2


def _bounded_ratio(numerator, denominator, cap):
    if denominator <= 0:
        if isinstance(numerator, pd.Series):
            return pd.Series(0.0, index=numerator.index)
        return 0.0
    ratio = numerator / denominator
    if isinstance(ratio, pd.Series):
        return ratio.clip(-cap, cap) / cap
    return max(min(ratio, cap), -cap) / cap


def _build_reason(row):
    reasons = []

    if row["avg_value_pct"] >= 12:
        reasons.append("priced below the model's fair-value estimate")
    elif row["avg_value_pct"] >= 5:
        reasons.append("slightly below the model's fair-value estimate")

    if row["median_mileage"] <= row["max_mileage_input"] * 0.75:
        reasons.append("lower mileage than your stated limit")

    if row["median_year"] >= row["min_year_input"] + 2:
        reasons.append("newer than your minimum year target")

    if row["confidence"] == "High":
        reasons.append("supported by many comparable sales")
    elif row["confidence"] == "Medium":
        reasons.append("supported by a reasonable number of comparable sales")

    if not reasons:
        reasons.append("a balanced match for your filters")

    return "; ".join(reasons)


def _load_data():
    data_path = Path(__file__).parent / "used_car_sales.csv"
    df = pd.read_csv(data_path)

    df["pricesold"] = pd.to_numeric(df["pricesold"], errors="coerce")
    df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Car_Age"] = pd.to_numeric(df["yearsold"], errors="coerce") - df["Year"]

    df = df[(df["Car_Age"] >= 0) & (df["Car_Age"].notna())]
    df = df[df["pricesold"].between(1000, 150000)]
    df = df[df["Mileage"].between(0, 300000)]
    df = df[df["Year"].between(1980, 2025)]
    df.dropna(subset=["pricesold", "Year", "Model", "Make"], inplace=True)
    df = df[~df["Model"].astype(str).str.strip().str.match(_BAD_MODEL_PATTERN)].copy()

    df["brand_clean"] = df["Make"].apply(_normalize_brand)
    df["brand_tier"] = df["brand_clean"].apply(_brand_tier)
    df["zip3"] = df["zipcode"].astype(str).str[:3]
    df["body_type_clean"] = df["BodyType"].apply(_normalize_body_type)
    df["drive_type_clean"] = df["DriveType"].apply(_normalize_drive_type)
    df["is_bad_record"] = df.apply(_has_bad_text, axis=1)

    df = df[~df["is_bad_record"]].copy()

    counts = df.groupby(["brand_clean", "Model"]).size().rename("_model_sample_count")
    df = df.join(counts, on=["brand_clean", "Model"])

    return df


_df = _load_data()


def recommend(
    budget: float,
    body_type: str = None,
    drive_type: str = None,
    make: str = None,
    max_mileage: int = None,
    min_year: int = None,
    zipcode: str = None,
    top_n: int = 10,
) -> list[dict]:
    """
    Recommend model-level options for a buyer.

    Returns aggregated recommendations, each representing a make/model family
    with typical pricing, mileage, year range, confidence, and explanation.
    """
    candidates = _df.copy()

    candidates = candidates[candidates["pricesold"] <= budget]

    if body_type:
        normalized_body = _normalize_body_type(body_type)
        candidates = candidates[candidates["body_type_clean"] == normalized_body]
    if drive_type:
        normalized_drive = _normalize_drive_type(drive_type)
        candidates = candidates[candidates["drive_type_clean"] == normalized_drive]
    if make:
        candidates = candidates[candidates["brand_clean"].str.lower() == make.lower()]
    if max_mileage:
        candidates = candidates[candidates["Mileage"] <= max_mileage]
    if min_year:
        candidates = candidates[candidates["Year"] >= min_year]

    if min_year is None:
        candidates = candidates[candidates["Year"] >= 2005]

    candidates = candidates[candidates["_model_sample_count"] >= 15]

    if len(candidates) == 0:
        return []

    scored_rows = []
    for _, row in candidates.iterrows():
        low, mid, high = get_price_range(
            make=row["Make"],
            model=row["Model"],
            year=int(row["Year"]),
            mileage=int(row["Mileage"]),
            body_type=row.get("BodyType", "Missing"),
            drive_type=row.get("DriveType", "Missing"),
            zipcode=str(row.get("zipcode", "Missing")),
            engine=str(row.get("Engine", "Missing")),
            trim=str(row.get("Trim", "Missing")),
        )

        actual = float(row["pricesold"])
        value_score = mid - actual
        value_pct = (value_score / mid * 100) if mid > 0 else 0.0

        if value_pct > 50 or value_pct < -35:
            continue

        scored_rows.append(
            {
                "make": row["Make"],
                "brand_clean": row["brand_clean"],
                "model": row["Model"],
                "body_type": row["body_type_clean"],
                "drive_type": row["drive_type_clean"],
                "year": int(row["Year"]),
                "mileage": int(row["Mileage"]),
                "actual_price": actual,
                "predicted_competitive": low,
                "predicted_fair": mid,
                "predicted_premium": high,
                "value_score": value_score,
                "value_pct": value_pct,
                "sample_count": int(row["_model_sample_count"]),
            }
        )

    if not scored_rows:
        return []

    scored_df = pd.DataFrame(scored_rows)

    grouped = (
        scored_df.groupby(["brand_clean", "model"], as_index=False)
        .agg(
            make=("make", "first"),
            body_type=("body_type", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            drive_type=("drive_type", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            year_min=("year", "min"),
            year_max=("year", "max"),
            median_year=("year", "median"),
            median_mileage=("mileage", "median"),
            typical_price=("actual_price", "median"),
            predicted_competitive=("predicted_competitive", "median"),
            predicted_fair=("predicted_fair", "median"),
            predicted_premium=("predicted_premium", "median"),
            avg_value_score=("value_score", "mean"),
            avg_value_pct=("value_pct", "mean"),
            candidate_count=("model", "size"),
            sample_count=("sample_count", "max"),
        )
    )

    effective_max_mileage = max_mileage if max_mileage else 120000
    effective_min_year = min_year if min_year else 2005

    grouped["value_component"] = grouped["avg_value_pct"].clip(-10, 20) / 20.0
    grouped["mileage_component"] = 1 - grouped["median_mileage"].clip(0, effective_max_mileage) / float(effective_max_mileage)
    grouped["year_component"] = _bounded_ratio(
        grouped["median_year"] - effective_min_year,
        max(2025 - effective_min_year, 1),
        1.0,
    )
    grouped["confidence_component"] = grouped["sample_count"].apply(_confidence_score)

    grouped["buyer_score"] = (
        0.45 * grouped["value_component"]
        + 0.20 * grouped["mileage_component"]
        + 0.15 * grouped["year_component"]
        + 0.20 * grouped["confidence_component"]
    )

    grouped = grouped[grouped["avg_value_pct"] > -5].copy()

    if grouped.empty:
        return []

    grouped["confidence"] = grouped["sample_count"].apply(_confidence_label)
    grouped["max_mileage_input"] = effective_max_mileage
    grouped["min_year_input"] = effective_min_year
    grouped["reason"] = grouped.apply(_build_reason, axis=1)

    grouped = grouped.sort_values(
        by=["buyer_score", "avg_value_pct", "sample_count"],
        ascending=False,
    )

    recommendations = []
    for _, row in grouped.head(top_n).iterrows():
        recommendations.append(
            {
                "make": row["make"],
                "model": row["model"],
                "title": "{} {}".format(row["make"], row["model"]),
                "body_type": row["body_type"],
                "drive_type": row["drive_type"],
                "year_range": "{}-{}".format(int(row["year_min"]), int(row["year_max"])),
                "typical_year": int(round(row["median_year"])),
                "typical_mileage": int(round(row["median_mileage"])),
                "typical_price": round(float(row["typical_price"]), 2),
                "predicted_competitive": round(float(row["predicted_competitive"]), 2),
                "predicted_fair": round(float(row["predicted_fair"]), 2),
                "predicted_premium": round(float(row["predicted_premium"]), 2),
                "avg_value_score": round(float(row["avg_value_score"]), 2),
                "avg_value_pct": round(float(row["avg_value_pct"]), 1),
                "buyer_score": round(float(row["buyer_score"]), 3),
                "sample_count": int(row["sample_count"]),
                "candidate_count": int(row["candidate_count"]),
                "confidence": row["confidence"],
                "reason": row["reason"],
            }
        )

    return recommendations


if __name__ == "__main__":
    print("Finding top 10 SUVs under $20,000...\n")
    results = recommend(budget=20000, body_type="SUV", top_n=10)

    for i, r in enumerate(results, 1):
        print("{}. {} ({})".format(i, r["title"], r["year_range"]))
        print(
            "   Typical price: ${:,.0f} | Fair value: ${:,.0f} | Value gap: ${:,.0f} ({:+.0f}%)".format(
                r["typical_price"], r["predicted_fair"], r["avg_value_score"], r["avg_value_pct"]
            )
        )
        print(
            "   Mileage: {:,} | Confidence: {} ({} comps)".format(
                r["typical_mileage"], r["confidence"], r["sample_count"]
            )
        )
        print("   Why: {}\n".format(r["reason"]))
