"""
Direction 2: Buyer Recommendation.

Recommend individual historical sale records and compare their observed sale
price against the Direction 1 fair-value estimate.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

try:
    from .predict import (
        _brand_tier,
        _normalize_brand,
        _model_q25,
        _model_q50,
        _model_q75,
        CPI_ADJUSTMENT,
    )
except ImportError:
    from predict import (
        _brand_tier,
        _normalize_brand,
        _model_q25,
        _model_q50,
        _model_q75,
        CPI_ADJUSTMENT,
    )


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


def _confidence_help(sample_count):
    if sample_count >= 100:
        return "High data support"
    if sample_count >= 30:
        return "Moderate data support"
    return "Limited data support"


def _build_reason(value_pct, sample_count, mileage, max_mileage, year, min_year):
    reasons = []
    if value_pct >= 18:
        reasons.append("priced well below the model's fair-value estimate")
    elif value_pct >= 8:
        reasons.append("priced below the model's fair-value estimate")

    if max_mileage is not None and mileage <= max_mileage * 0.75:
        reasons.append("lower mileage than your stated limit")

    if min_year is not None and year >= min_year + 2:
        reasons.append("newer than your minimum year target")

    confidence = _confidence_label(sample_count)
    if confidence == "High":
        reasons.append("supported by many comparable sales")
    elif confidence == "Medium":
        reasons.append("supported by a reasonable number of comparable sales")

    if not reasons:
        reasons.append("a solid match for your filters")

    return "; ".join(reasons)


def _predict_price_ranges_batch(candidates_df):
    batch = pd.DataFrame({
        "Mileage": candidates_df["Mileage"].astype(float),
        "Car_Age": candidates_df["Car_Age"].astype(float),
        "miles_per_year": candidates_df["Mileage"].astype(float) / (candidates_df["Car_Age"].astype(float) + 1),
        "Engine": candidates_df["Engine"].fillna("Missing").astype(str),
        "Trim": candidates_df["Trim"].fillna("Missing").astype(str),
        "DriveType": candidates_df["DriveType"].fillna("Missing").astype(str),
        "BodyType": candidates_df["BodyType"].fillna("Missing").astype(str),
        "brand_clean": candidates_df["brand_clean"].astype(str),
        "brand_tier": candidates_df["brand_tier"].astype(str),
        "Model": candidates_df["Model"].fillna("Missing").astype(str),
        "zip3": candidates_df["zip3"].astype(str),
    })

    lows = _model_q25.predict(batch) * CPI_ADJUSTMENT
    mids = _model_q50.predict(batch) * CPI_ADJUSTMENT
    highs = _model_q75.predict(batch) * CPI_ADJUSTMENT

    stacked = np.column_stack([lows, mids, highs])
    stacked.sort(axis=1)
    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


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
    Recommend individual historical sale records for a buyer.

    Each returned item is a single sold-car example, scored by how far below
    the Direction 1 fair-value estimate it appears.
    """
    candidates = _df.copy()

    candidates = candidates[candidates["pricesold"] <= budget]

    # Keep recommendations near the buyer's actual price range instead of
    # surfacing very cheap but irrelevant vehicles.
    if budget >= 15000:
        candidates = candidates[candidates["pricesold"] >= budget * 0.82]
    elif budget >= 8000:
        candidates = candidates[candidates["pricesold"] >= budget * 0.75]

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

    candidates = candidates.reset_index(drop=True)
    lows, mids, highs = _predict_price_ranges_batch(candidates)

    actuals = candidates["pricesold"].astype(float).to_numpy()
    value_scores = mids - actuals
    value_pcts = np.where(mids > 0, value_scores / mids * 100, 0.0)

    price_gap_from_budget = np.abs(actuals - float(budget))
    if budget > 0:
        budget_fit = 1.0 - np.clip(price_gap_from_budget / float(budget), 0, 1)
    else:
        budget_fit = np.zeros_like(actuals)

    mask = (value_pcts <= 35) & (value_pcts >= -20)

    recommendations = []
    for i in np.where(mask)[0]:
        row = candidates.iloc[i]
        actual = float(actuals[i])
        value_pct = float(value_pcts[i])
        value_score = float(value_scores[i])
        sample_count = int(row["_model_sample_count"])
        confidence = _confidence_label(sample_count)
        reason = _build_reason(
            value_pct=value_pct,
            sample_count=sample_count,
            mileage=int(row["Mileage"]),
            max_mileage=max_mileage,
            year=int(row["Year"]),
            min_year=min_year,
        )

        normalized_value = min(max(value_pct, 0.0), 25.0) / 25.0
        buyer_score = 0.50 * normalized_value + 0.50 * float(budget_fit[i])

        recommendations.append(
            {
                "id": int(row["ID"]) if not pd.isna(row.get("ID")) else None,
                "make": row["Make"],
                "model": row["Model"],
                "title": "{} {} {}".format(int(row["Year"]), row["Make"], row["Model"]),
                "body_type": row["body_type_clean"],
                "drive_type": row["drive_type_clean"],
                "year_range": str(int(row["Year"])),
                "typical_year": int(row["Year"]),
                "typical_mileage": int(row["Mileage"]),
                "listing_price": round(actual, 2),
                "typical_price": round(actual, 2),
                "predicted_competitive": round(float(lows[i]), 2),
                "predicted_fair": round(float(mids[i]), 2),
                "predicted_premium": round(float(highs[i]), 2),
                "avg_value_score": round(value_score, 2),
                "avg_value_pct": round(value_pct, 1),
                "buyer_score": round(buyer_score, 3),
                "budget_fit": round(float(budget_fit[i]), 3),
                "sample_count": sample_count,
                "candidate_count": 1,
                "confidence": confidence,
                "confidence_help": _confidence_help(sample_count),
                "data_support": _confidence_help(sample_count),
                "reason": reason,
                "trim": "" if pd.isna(row.get("Trim")) else str(row.get("Trim")),
                "engine": "" if pd.isna(row.get("Engine")) else str(row.get("Engine")),
                "zipcode": "" if pd.isna(row.get("zipcode")) else str(row.get("zipcode")),
            }
        )

    recommendations.sort(
        key=lambda r: (r["buyer_score"], r["avg_value_pct"], r["listing_price"]),
        reverse=True,
    )

    return recommendations[:top_n]


if __name__ == "__main__":
    print("Finding top 10 SUVs under $20,000...\n")
    results = recommend(budget=20000, body_type="SUV", top_n=10)

    for i, r in enumerate(results, 1):
        print("{}. {}".format(i, r["title"]))
        print(
            "   Sold price: ${:,.0f} | Fair value: ${:,.0f} | Value gap: ${:,.0f} ({:+.0f}%)".format(
                r["typical_price"], r["predicted_fair"], r["avg_value_score"], r["avg_value_pct"]
            )
        )
        print(
            "   Mileage: {:,} | Confidence: {} ({} comps)".format(
                r["typical_mileage"], r["confidence"], r["sample_count"]
            )
        )
        print("   Why: {}\n".format(r["reason"]))
