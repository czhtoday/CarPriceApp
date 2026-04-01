"""
Benchmark multiple quantile-regression models on the used-car dataset.

Compares:
- CatBoostRegressor (native categorical handling)
- LightGBMRegressor (native categorical handling)
- XGBRegressor (native categorical handling)
- HistGradientBoostingRegressor (ordinal-encoded categorical baseline)

Outputs a CSV with one row per model and prints a sorted summary.
Can also run K-fold cross-validation and save fold-level results.

Example:
    python benchmark_quantile_models.py
    python benchmark_quantile_models.py --sample-size 30000
    python benchmark_quantile_models.py --cv-folds 5
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor


RANDOM_STATE = 42
QUANTILES = [(0.25, "q25"), (0.50, "q50"), (0.75, "q75")]


brand_tier_map = {
    "Mercedes-Benz": "Luxury",
    "BMW": "Luxury",
    "Audi": "Luxury",
    "Lexus": "Luxury",
    "Porsche": "Luxury",
    "Jaguar": "Luxury",
    "Land Rover": "Luxury",
    "Cadillac": "Luxury",
    "Volvo": "Luxury",
    "Acura": "Luxury",
    "Infiniti": "Luxury",
    "Lincoln": "Luxury",
    "Maserati": "Luxury",
    "Bentley": "Luxury",
    "Tesla": "Luxury",
    "Alfa Romeo": "Luxury",
    "Hummer": "Luxury",
    "Toyota": "Mid",
    "Honda": "Mid",
    "Volkswagen": "Mid",
    "Subaru": "Mid",
    "Mazda": "Mid",
    "Hyundai": "Mid",
    "Kia": "Mid",
    "Buick": "Mid",
    "Mini": "Mid",
    "GMC": "Mid",
    "Chrysler": "Mid",
    "Nissan": "Mid",
    "Ram": "Mid",
    "Jeep": "Mid",
    "Dodge": "Mid",
    "Ford": "Mid",
    "Chevrolet": "Mid",
    "Fiat": "Economy",
    "Mitsubishi": "Economy",
    "Pontiac": "Economy",
    "Oldsmobile": "Economy",
    "Plymouth": "Economy",
    "Saab": "Economy",
    "Saturn": "Economy",
    "Mercury": "Economy",
    "AMC": "Economy",
    "MG": "Economy",
    "Triumph": "Economy",
    "Datsun": "Economy",
    "Studebaker": "Economy",
    "International Harvester": "Economy",
    "Other Makes": "Economy",
    "Replica/Kit Makes": "Economy",
}

variation_map = {
    "Chevy": "Chevrolet",
    "Cheverolet": "Chevrolet",
    "Volkswagon": "Volkswagen",
    "Pontlac": "Pontiac",
    "Ponitac": "Pontiac",
    "Piymouth": "Plymouth",
    "Crhysler": "Chrysler",
    "Chrylser": "Chrysler",
    "Infinity": "Infiniti",
    "Suburu": "Subaru",
    "Mercedes": "Mercedes-Benz",
    "Internatonal": "International Harvester",
    "Vw": "Volkswagen",
    "Vw/Other": "Volkswagen",
    "White/Gmc": "GMC",
    "Gl550": "Mercedes-Benz",
}

KNOWN_BRANDS = list(brand_tier_map.keys())


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s\-]+", " ", s)
    return s


KNOWN_BRANDS_NORM = {_norm(b): b for b in KNOWN_BRANDS}


def normalize_brand(raw: object) -> str | None:
    if pd.isna(raw):
        return None
    s_norm = _norm(str(raw))
    if s_norm in KNOWN_BRANDS_NORM:
        return KNOWN_BRANDS_NORM[s_norm]
    for kb_norm, kb_canon in KNOWN_BRANDS_NORM.items():
        if len(kb_norm.split()) > 1 and s_norm.startswith(kb_norm + " "):
            return kb_canon
    first_word = s_norm.split()[0]
    for kb_norm, kb_canon in KNOWN_BRANDS_NORM.items():
        if kb_norm == first_word:
            return kb_canon
    return first_word.title()


def brand_to_tier(clean_brand: str | None) -> str:
    if clean_brand is None:
        return "Economy"
    canonical = variation_map.get(clean_brand, clean_brand)
    return brand_tier_map.get(canonical, "Economy")


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    residual = y_true - y_pred
    return float(
        np.mean(np.where(residual >= 0, alpha * residual, (alpha - 1.0) * residual))
    )


def clean_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Car_Age"] = pd.to_numeric(df["yearsold"], errors="coerce") - pd.to_numeric(
        df["Year"], errors="coerce"
    )
    df = df[(df["Car_Age"] >= 0) & (df["Car_Age"].notna())].copy()

    df["pricesold"] = pd.to_numeric(df["pricesold"], errors="coerce")
    df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
    df = df[df["pricesold"].between(1000, 150000)]
    df = df[df["Mileage"].between(0, 300000)]

    df["brand_clean"] = df["Make"].apply(normalize_brand)
    df["brand_tier"] = df["brand_clean"].apply(brand_to_tier)
    df["zip3"] = df["zipcode"].astype(str).str[:3]
    df["miles_per_year"] = df["Mileage"] / (df["Car_Age"] + 1)
    df.dropna(subset=["pricesold"], inplace=True)
    return df


@dataclass(frozen=True)
class FeatureSpec:
    num_cols: list[str]
    cat_cols: list[str]


FEATURE_SPEC = FeatureSpec(
    num_cols=["Mileage", "Car_Age", "miles_per_year"],
    cat_cols=[
        "Engine",
        "Trim",
        "DriveType",
        "BodyType",
        "brand_clean",
        "brand_tier",
        "Model",
        "zip3",
    ],
)


def build_feature_frame(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, np.ndarray]:
    feature_cols = spec.num_cols + spec.cat_cols
    X = df[feature_cols].reset_index(drop=True).copy()
    y = df["pricesold"].to_numpy()
    return X, y


def fill_missing(
    X_train: pd.DataFrame, X_test: pd.DataFrame, spec: FeatureSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in spec.cat_cols:
        X_train[col] = X_train[col].fillna("Missing").astype(str)
        X_test[col] = X_test[col].fillna("Missing").astype(str)

    for col in spec.num_cols:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    return X_train, X_test


def to_category_dtype(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cols: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)
    return X_train, X_test


def train_predict_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    alpha: float,
    spec: FeatureSpec,
) -> np.ndarray:
    model = CatBoostRegressor(
        loss_function=f"Quantile:alpha={alpha}",
        iterations=300,
        depth=6,
        learning_rate=0.1,
        verbose=0,
        random_seed=RANDOM_STATE,
    )
    model.fit(X_train, y_train, cat_features=spec.cat_cols)
    return model.predict(X_test)


def train_predict_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    alpha: float,
    spec: FeatureSpec,
) -> np.ndarray:
    X_train_cat, X_test_cat = to_category_dtype(X_train, X_test, spec.cat_cols)
    model = LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X_train_cat, y_train, categorical_feature=spec.cat_cols)
    return model.predict(X_test_cat)


def train_predict_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    alpha: float,
    spec: FeatureSpec,
) -> np.ndarray:
    X_train_cat, X_test_cat = to_category_dtype(X_train, X_test, spec.cat_cols)
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train_cat, y_train)
    return model.predict(X_test_cat)


def train_predict_histgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    alpha: float,
    spec: FeatureSpec,
) -> np.ndarray:
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                spec.cat_cols,
            ),
            ("num", "passthrough", spec.num_cols),
        ],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=alpha,
                    learning_rate=0.05,
                    max_iter=300,
                    max_depth=6,
                    min_samples_leaf=20,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


MODEL_FNS = {
    "CatBoost": train_predict_catboost,
    "LightGBM": train_predict_lightgbm,
    "XGBoost": train_predict_xgboost,
    "HistGB": train_predict_histgb,
}


def evaluate_predictions(model_name: str, y_true: np.ndarray, preds: Dict[str, np.ndarray]) -> dict[str, float | str]:
    q25 = preds["q25"]
    q50 = preds["q50"]
    q75 = preds["q75"]

    lower = np.minimum(q25, q75)
    upper = np.maximum(q25, q75)
    widths = upper - lower
    coverage = ((y_true >= lower) & (y_true <= upper)).mean()
    crossing_rate = ((q25 > q50) | (q50 > q75) | (q25 > q75)).mean()

    return {
        "model": model_name,
        "q50_mae": float(mean_absolute_error(y_true, q50)),
        "pinball_q25": pinball_loss(y_true, q25, 0.25),
        "pinball_q50": pinball_loss(y_true, q50, 0.50),
        "pinball_q75": pinball_loss(y_true, q75, 0.75),
        "coverage_q25_q75": float(coverage),
        "avg_width_q25_q75": float(np.mean(widths)),
        "median_width_q25_q75": float(np.median(widths)),
        "crossing_rate": float(crossing_rate),
    }


def summarize_results(results: list[dict[str, float | str]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    metric_cols = [c for c in df.columns if c not in {"model", "fold"}]
    summary = df.groupby("model")[metric_cols].agg(["mean", "std"])
    summary.columns = [
        f"{col}_{stat}" if stat != "mean" else col
        for col, stat in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index()
    return summary.sort_values(
        ["q50_mae", "pinball_q50", "coverage_q25_q75"], ascending=[True, True, False]
    )


def run_benchmark(
    data_path: str,
    results_path: str,
    sample_size: int | None,
    test_size: float,
    cv_folds: int,
) -> pd.DataFrame:
    df = clean_dataset(data_path)
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

    X, y = build_feature_frame(df, FEATURE_SPEC)
    results = []

    if cv_folds > 1:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        splits = list(splitter.split(X))
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )
        splits = [(X_train.index.to_numpy(), X_test.index.to_numpy())]

    for model_name, predict_fn in MODEL_FNS.items():
        print(f"\n=== {model_name} ===", flush=True)
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = y[train_idx]
            y_test = y[test_idx]
            X_train, X_test = fill_missing(X_train, X_test, FEATURE_SPEC)

            preds: Dict[str, np.ndarray] = {}
            t0 = time.time()
            for alpha, label in QUANTILES:
                start = time.time()
                preds[label] = predict_fn(X_train, y_train, X_test, alpha, FEATURE_SPEC)
                elapsed = time.time() - start
                fold_prefix = f"fold {fold_idx}/{len(splits)}" if cv_folds > 1 else "holdout"
                print(f"  {fold_prefix} trained {label} in {elapsed:.1f}s", flush=True)

            metrics = evaluate_predictions(model_name, y_test, preds)
            metrics["train_time_sec"] = float(time.time() - t0)
            metrics["fold"] = fold_idx
            results.append(metrics)
            fold_prefix = f"fold {fold_idx}/{len(splits)}" if cv_folds > 1 else "holdout"
            print(
                "  {} q50 MAE=${:,.0f}, coverage={:.1%}, avg width=${:,.0f}, crossing={:.1%}".format(
                    fold_prefix,
                    metrics["q50_mae"],
                    metrics["coverage_q25_q75"],
                    metrics["avg_width_q25_q75"],
                    metrics["crossing_rate"],
                ),
                flush=True,
            )

    results_df = summarize_results(results)
    results_df.to_csv(results_path, index=False)

    if cv_folds > 1:
        folds_path = results_path.replace(".csv", "_folds.csv")
        pd.DataFrame(results).to_csv(folds_path, index=False)
        print("\nSaved fold-level results to", folds_path)

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="used_car_sales.csv")
    parser.add_argument("--results-path", default="quantile_benchmark_results.csv")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_df = run_benchmark(
        data_path=args.data_path,
        results_path=args.results_path,
        sample_size=args.sample_size,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
    )

    print("\nSaved results to", args.results_path)
    display_cols = [
        "model",
        "q50_mae",
        "q50_mae_std",
        "pinball_q25",
        "pinball_q25_std",
        "pinball_q50",
        "pinball_q50_std",
        "pinball_q75",
        "pinball_q75_std",
        "coverage_q25_q75",
        "coverage_q25_q75_std",
        "avg_width_q25_q75",
        "avg_width_q25_q75_std",
        "median_width_q25_q75",
        "median_width_q25_q75_std",
        "crossing_rate",
        "crossing_rate_std",
        "train_time_sec",
        "train_time_sec_std",
    ]
    print(results_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
