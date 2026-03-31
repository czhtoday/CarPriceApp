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
# Region Analysis
# ================================
def compute_region_analysis(df, predict_func):
    df_region = df.copy()

    df_region["pred_price"] = df_region.apply(
        lambda row: predict_func(row), axis=1
    )

    df_region = df_region[df_region["pred_price"].notna()].copy()

    df_region["residual"] = (
        df_region["pricesold"] - df_region["pred_price"]
    )

    region_analysis = (
        df_region.groupby("zip3")["residual"]
        .mean()
        .reset_index()
        .sort_values("residual")
    )

    return region_analysis


def get_region_deals(predict_func):
    df = load_data()
    region_analysis = compute_region_analysis(df, predict_func)

    return region_analysis.head(10)