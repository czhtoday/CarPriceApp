"""
Price range prediction for used cars.

Usage:
    from predict import get_price_range

    low, mid, high = get_price_range(
        make="Toyota",
        model="RAV4",
        year=2018,
        mileage=45000,
        body_type="SUV",
        drive_type="AWD",
        zipcode="900",       # first 3 digits, or full zip (truncated automatically)
        engine="2.5L I4",    # optional
        trim="XLE",          # optional
    )
    # low  = competitive price (25th percentile)
    # mid  = fair market value (50th percentile)
    # high = premium price (75th percentile)
"""

import re
import json
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from pathlib import Path

_DIR = Path(__file__).parent

# ── Load models (once on import) ──────────────────────────────────────────

_model_q25 = CatBoostRegressor().load_model(str(_DIR / "model_q25.cbm"))
_model_q50 = CatBoostRegressor().load_model(str(_DIR / "model_q50.cbm"))
_model_q75 = CatBoostRegressor().load_model(str(_DIR / "model_q75.cbm"))

with open(_DIR / "feature_medians.json") as f:
    _medians = json.load(f)

# ── Brand helpers (same as training) ──────────────────────────────────────

_brand_tier_map = {
    "Mercedes-Benz": "Luxury", "BMW": "Luxury", "Audi": "Luxury",
    "Lexus": "Luxury", "Porsche": "Luxury", "Jaguar": "Luxury",
    "Land Rover": "Luxury", "Cadillac": "Luxury", "Volvo": "Luxury",
    "Acura": "Luxury", "Infiniti": "Luxury", "Lincoln": "Luxury",
    "Maserati": "Luxury", "Bentley": "Luxury", "Tesla": "Luxury",
    "Alfa Romeo": "Luxury", "Hummer": "Luxury",
    "Toyota": "Mid", "Honda": "Mid", "Volkswagen": "Mid",
    "Subaru": "Mid", "Mazda": "Mid", "Hyundai": "Mid",
    "Kia": "Mid", "Buick": "Mid", "Mini": "Mid",
    "GMC": "Mid", "Chrysler": "Mid", "Nissan": "Mid",
    "Ram": "Mid", "Jeep": "Mid", "Dodge": "Mid",
    "Ford": "Mid", "Chevrolet": "Mid",
    "Fiat": "Economy", "Mitsubishi": "Economy", "Pontiac": "Economy",
    "Oldsmobile": "Economy", "Plymouth": "Economy", "Saab": "Economy",
    "Saturn": "Economy", "Mercury": "Economy", "AMC": "Economy",
    "MG": "Economy", "Triumph": "Economy", "Datsun": "Economy",
    "Studebaker": "Economy", "International Harvester": "Economy",
    "Other Makes": "Economy", "Replica/Kit Makes": "Economy",
}

_variation_map = {
    "Chevy": "Chevrolet", "Cheverolet": "Chevrolet",
    "Volkswagon": "Volkswagen", "Pontlac": "Pontiac",
    "Ponitac": "Pontiac", "Piymouth": "Plymouth",
    "Crhysler": "Chrysler", "Chrylser": "Chrysler",
    "Infinity": "Infiniti", "Suburu": "Subaru",
    "Mercedes": "Mercedes-Benz", "Internatonal": "International Harvester",
    "Vw": "Volkswagen", "Vw/Other": "Volkswagen",
    "White/Gmc": "GMC", "Gl550": "Mercedes-Benz",
}

_KNOWN_BRANDS_NORM = {
    re.sub(r'[\s\-]+', ' ', b.strip().lower()): b
    for b in _brand_tier_map
}

def _normalize_brand(raw):
    if raw is None:
        return "Missing"
    s = re.sub(r'[\s\-]+', ' ', str(raw).strip().lower())
    if s in _KNOWN_BRANDS_NORM:
        return _KNOWN_BRANDS_NORM[s]
    for kb_norm, kb_canon in _KNOWN_BRANDS_NORM.items():
        if len(kb_norm.split()) > 1 and s.startswith(kb_norm + " "):
            return kb_canon
    first_word = s.split()[0]
    for kb_norm, kb_canon in _KNOWN_BRANDS_NORM.items():
        if kb_norm == first_word:
            return kb_canon
    return first_word.title()

def _brand_tier(clean_brand):
    canonical = _variation_map.get(clean_brand, clean_brand)
    return _brand_tier_map.get(canonical, "Economy")


# ── Public API ────────────────────────────────────────────────────────────

from datetime import datetime
CURRENT_YEAR = datetime.now().year

def get_price_range(
    make: str,
    model: str,
    year: int,
    mileage: int,
    body_type: str,
    drive_type: str,
    zipcode: str = "Missing",
    engine: str = "Missing",
    trim: str = "Missing",
) -> tuple[float, float, float]:
    """
    Predict the fair market price range for a used car.

    Returns:
        (low, mid, high) — 25th, 50th, and 75th percentile price estimates.
          - low  = competitive price (sells fast)
          - mid  = fair market value
          - high = premium price (might sit longer)
        The seller's asking price can be compared against these three points
        on a gauge.
    """
    brand_clean = _normalize_brand(make)
    brand_tier = _brand_tier(brand_clean)
    zip3 = str(zipcode)[:3] if zipcode else "Missing"
    car_age = max(CURRENT_YEAR - int(year), 0)
    miles_per_year = mileage / (car_age + 1)

    row = pd.DataFrame([{
        "Mileage": float(mileage),
        "Car_Age": float(car_age),
        "miles_per_year": float(miles_per_year),
        "Engine": str(engine) if engine else "Missing",
        "Trim": str(trim) if trim else "Missing",
        "DriveType": str(drive_type) if drive_type else "Missing",
        "BodyType": str(body_type) if body_type else "Missing",
        "brand_clean": brand_clean,
        "brand_tier": brand_tier,
        "Model": str(model) if model else "Missing",
        "zip3": zip3,
    }])

    low = float(_model_q25.predict(row)[0])
    mid = float(_model_q50.predict(row)[0])
    high = float(_model_q75.predict(row)[0])

    # Ensure ordering (safety check)
    low, mid, high = sorted([low, mid, high])

    return round(low, 2), round(mid, 2), round(high, 2)


if __name__ == "__main__":
    # Quick smoke test
    low, mid, high = get_price_range(
        make="Toyota", model="RAV4", year=2018, mileage=45000,
        body_type="SUV", drive_type="AWD", zipcode="90210",
    )
    print(f"Toyota RAV4 2018 (45k mi):")
    print(f"  Competitive: ${low:,.0f}  |  Fair: ${mid:,.0f}  |  Premium: ${high:,.0f}")

    low, mid, high = get_price_range(
        make="BMW", model="X5", year=2016, mileage=80000,
        body_type="SUV", drive_type="AWD", zipcode="10001",
    )
    print(f"BMW X5 2016 (80k mi):")
    print(f"  Competitive: ${low:,.0f}  |  Fair: ${mid:,.0f}  |  Premium: ${high:,.0f}")

    low, mid, high = get_price_range(
        make="Honda", model="Civic", year=2020, mileage=30000,
        body_type="Sedan", drive_type="FWD", zipcode="60601",
        trim="EX", engine="2.0L I4",
    )
    print(f"Honda Civic 2020 (30k mi, EX trim):")
    print(f"  Competitive: ${low:,.0f}  |  Fair: ${mid:,.0f}  |  Premium: ${high:,.0f}")
