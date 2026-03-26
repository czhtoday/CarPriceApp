"""
Direction 2: Buyer Recommendation — find the best-value cars matching buyer preferences.

Usage:
    from buyer_recommend import recommend

    results = recommend(
        budget=20000,
        body_type="SUV",          # optional
        drive_type="AWD",         # optional
        make="Toyota",            # optional
        max_mileage=80000,        # optional
        min_year=2015,            # optional
        zipcode="90210",          # optional
        top_n=10,                 # default 10
    )

    # Returns a list of dicts, ranked by value score (best deals first).
"""

import pandas as pd
import numpy as np
from predict import get_price_range, _normalize_brand, _brand_tier, _model_q25, _model_q50, _model_q75

# ── Load and clean dataset (same cleaning as training) ─────────────────────

def _load_data():
    from pathlib import Path
    import re

    # Reuse brand helpers from predict module
    from predict import _normalize_brand, _brand_tier

    df = pd.read_csv(Path(__file__).parent / "used_car_sales.csv")

    df["pricesold"] = pd.to_numeric(df["pricesold"], errors="coerce")
    df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
    df["Car_Age"] = pd.to_numeric(df["yearsold"], errors="coerce") - pd.to_numeric(df["Year"], errors="coerce")

    df = df[(df["Car_Age"] >= 0) & (df["Car_Age"].notna())]
    df = df[df["pricesold"].between(1000, 150000)]
    df = df[df["Mileage"].between(0, 300000)]
    df.dropna(subset=["pricesold"], inplace=True)

    df["brand_clean"] = df["Make"].apply(_normalize_brand)
    df["brand_tier"] = df["brand_clean"].apply(_brand_tier)
    df["zip3"] = df["zipcode"].astype(str).str[:3]

    # Count listings per make+model — used to filter out rare cars the model
    # can't predict reliably
    counts = df.groupby(["brand_clean", "Model"]).size().rename("_listing_count")
    df = df.join(counts, on=["brand_clean", "Model"])

    return df

_df = _load_data()


# ── Batch prediction ──────────────────────────────────────────────────────


def _get_price_ranges_batch(candidates_df):
    """Predict price ranges for all candidates at once (one model call per quantile)."""
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

    lows = _model_q25.predict(batch)
    mids = _model_q50.predict(batch)
    highs = _model_q75.predict(batch)

    # Ensure ordering per row
    stacked = np.column_stack([lows, mids, highs])
    stacked.sort(axis=1)

    return stacked[:, 0], stacked[:, 1], stacked[:, 2]


# ── Public API ─────────────────────────────────────────────────────────────

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
    Find the best-value cars within the buyer's constraints.

    Returns a list of dicts sorted by value (best deals first), each containing:
      - car details (make, model, year, mileage, etc.)
      - predicted fair price range (q25, q50, q75)
      - value_score (how much below fair market value)
      - recommendation reason
    """
    candidates = _df.copy()

    # ── Apply filters ──────────────────────────────────────────────────────
    candidates = candidates[candidates["pricesold"] <= budget]

    if body_type:
        candidates = candidates[candidates["BodyType"].str.lower() == body_type.lower()]
    if drive_type:
        candidates = candidates[candidates["DriveType"].str.lower() == drive_type.lower()]
    if make:
        candidates = candidates[candidates["brand_clean"].str.lower() == make.lower()]
    if max_mileage:
        candidates = candidates[candidates["Mileage"] <= max_mileage]
    if min_year:
        candidates = candidates[candidates["Year"] >= min_year]

    # Only recommend cars with enough comparable listings
    candidates = candidates[candidates["_listing_count"] >= 30]

    # Default: exclude very old cars (pre-2005) unless the buyer specifically
    # set min_year — old cars have unreliable predictions
    if min_year is None:
        candidates = candidates[candidates["Year"] >= 2005]

    if len(candidates) == 0:
        return []

    # ── Score all candidates in batch ────────────────────────────────────────
    candidates = candidates.reset_index(drop=True)
    lows, mids, highs = _get_price_ranges_batch(candidates)
    actuals = candidates["pricesold"].values

    value_scores = mids - actuals
    value_pcts = np.where(mids > 0, value_scores / mids * 100, 0)

    # Filter outliers where model prediction is unreliable
    mask = value_pcts <= 50

    scored = []
    for i in np.where(mask)[0]:
        row = candidates.iloc[i]
        vp = value_pcts[i]

        if vp > 15:
            reason = "Priced well below market value"
        elif vp > 5:
            reason = "Priced below market value"
        elif vp > -5:
            reason = "Fairly priced"
        else:
            reason = "Priced above market value"

        scored.append({
            "make": row["Make"],
            "model": row["Model"],
            "year": int(row["Year"]),
            "mileage": int(row["Mileage"]),
            "body_type": row.get("BodyType", ""),
            "drive_type": row.get("DriveType", ""),
            "trim": row.get("Trim", ""),
            "actual_price": round(actuals[i], 2),
            "predicted_competitive": round(float(lows[i]), 2),
            "predicted_fair": round(float(mids[i]), 2),
            "predicted_premium": round(float(highs[i]), 2),
            "value_score": round(float(value_scores[i]), 2),
            "value_pct": round(float(vp), 1),
            "reason": reason,
        })

    # Sort by value score descending (best deals first)
    scored.sort(key=lambda x: x["value_score"], reverse=True)

    return scored[:top_n]


if __name__ == "__main__":
    print("Finding top 10 SUVs under $20,000...\n")
    results = recommend(budget=20000, body_type="SUV", top_n=10)

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['year']} {r['make']} {r['model']} ({r['trim']})")
        print(f"   Listed: ${r['actual_price']:,.0f}  |  Fair value: ${r['predicted_fair']:,.0f}  |  Savings: ${r['value_score']:,.0f} ({r['value_pct']:+.0f}%)")
        print(f"   Mileage: {r['mileage']:,}  |  {r['reason']}")
        print()
