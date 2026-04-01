import pandas as pd


# ================================
# Load data
# ================================
def load_data(path="used_car_sales.csv"):
    df = pd.read_csv(path)

    # basic cleaning
    df = df[df["pricesold"] > 0]
    df = df[df["Mileage"] > 0]
    df = df[df["Year"] > 1990]
    df = df[df["Mileage"] < 300000]

    # feature engineering
    df["Car_Age"] = 2019 - df["Year"]
    df = df[df["Car_Age"] >= 0]

    df["miles_per_year"] = df["Mileage"] / (df["Car_Age"] + 1)
    df["zip3"] = df["zipcode"].fillna("Missing").astype(str).str[:3]

    return df


# ================================
# Depreciation Analysis
# ================================
def compute_depreciation(df):
    avg_price = df.groupby("Car_Age")["pricesold"].mean().reset_index()

    avg_price["price_drop"] = avg_price["pricesold"].diff()
    avg_price["depreciation"] = -avg_price["price_drop"]

    return avg_price


def estimate_future_drop(avg_price, age, years=2):
    future = avg_price[
        avg_price["Car_Age"].between(age, age + years)
    ]
    return future["depreciation"].sum()


# ================================
# Region Analysis Helpers
# ================================
def map_region(zip3: str) -> str:
    """
    Map zip3 into a broad readable region.
    """
    z = str(zip3).strip()

    if len(z) == 0 or z.lower() == "missing":
        return "Unknown"

    if not z.isdigit():
        return "Other / Non-US"

    z = int(z)

    if 0 <= z <= 199:
        return "Northeast"
    elif 200 <= z <= 399:
        return "South"
    elif 400 <= z <= 599:
        return "Midwest"
    elif 600 <= z <= 799:
        return "Central / Plains"
    elif 800 <= z <= 899:
        return "Mountain West"
    elif 900 <= z <= 999:
        return "West Coast"
    else:
        return "Other"


def deal_label(price_advantage: float) -> str:
    """
    Convert price advantage into an intuitive user-facing label.
    """
    if price_advantage > 3000:
        return "🔥 Best Deals"
    elif price_advantage > 2500:
        return "Good Deals"
    elif price_advantage > 0:
        return "Slightly Cheaper"
    else:
        return "Expensive"


# ================================
# Region Analysis
# ================================
def compute_region_analysis(df, predict_func):
    df_region = df.copy()

    # predicted fair price from model
    df_region["pred_price"] = df_region.apply(
        lambda row: predict_func(row), axis=1
    )

    df_region = df_region[df_region["pred_price"].notna()].copy()

    # residual: actual - predicted
    df_region["residual"] = (
        df_region["pricesold"] - df_region["pred_price"]
    )

    # aggregate first by zip3
    zip3_analysis = (
        df_region.groupby("zip3")
        .agg(
            avg_residual=("residual", "mean"),
            sample_size=("residual", "size"),
        )
        .reset_index()
    )

    # map zip3 to broader region
    zip3_analysis["region"] = zip3_analysis["zip3"].apply(map_region)

    # aggregate to readable user-facing region
    region_analysis = (
        zip3_analysis.groupby("region")
        .agg(
            avg_residual=("avg_residual", "mean"),
            sample_size=("sample_size", "sum"),
        )
        .reset_index()
    )

    # remove small groups for stability
    region_analysis = region_analysis[region_analysis["sample_size"] >= 20].copy()

    # turn residual into user-facing metric
    # negative residual means cheaper than expected
    region_analysis["price_advantage"] = -region_analysis["avg_residual"]

    # add human-readable label
    region_analysis["deal_label"] = region_analysis["price_advantage"].apply(deal_label)

    # rank by best savings first
    region_analysis = region_analysis.sort_values(
        "price_advantage", ascending=False
    ).reset_index(drop=True)

    return region_analysis


def get_region_deals(predict_func, top_n=6):
    """
    Return top undervalued regions for UI display.

    Output columns:
    - region
    - avg_residual
    - sample_size
    - price_advantage
    - deal_label
    """
    df = load_data()
    region_analysis = compute_region_analysis(df, predict_func)

    return region_analysis.head(top_n)