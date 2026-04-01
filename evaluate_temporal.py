"""
Time-aware model evaluation for CarPriceApp.

Trains on 2018-2019 sales, tests on 2020 sales.
This gives honest out-of-sample metrics that reflect real-world performance,
directly addressing TA feedback on validation strategy.
"""

import re
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_DIR = Path(__file__).parent

# ── Brand helpers (same as predict.py) ───────────────────────────────────
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

_BAD_TEXT_PATTERN = re.compile(
    r"auction|no reserve|salvage|rebuilt|parts only|parts car|rebuilt title|"
    r"bill of sale|not running|project car",
    re.IGNORECASE,
)
_BAD_MODEL_PATTERN = re.compile(r"^other\b|^nan$", re.IGNORECASE)


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


TRAINING_ERA_YEAR = 2019
FEATURES = [
    "Mileage", "Car_Age", "miles_per_year",
    "Engine", "Trim", "DriveType", "BodyType",
    "brand_clean", "brand_tier", "Model", "zip3",
]
CAT_FEATURES = [
    "Engine", "Trim", "DriveType", "BodyType",
    "brand_clean", "brand_tier", "Model", "zip3",
]

# ── Load & prep data ────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(_DIR / "used_car_sales.csv")

df["pricesold"] = pd.to_numeric(df["pricesold"], errors="coerce")
df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["yearsold"] = pd.to_numeric(df["yearsold"], errors="coerce")
df["Car_Age"] = df["yearsold"] - df["Year"]

# Apply same filters as training
df = df[(df["Car_Age"] >= 0) & (df["Car_Age"].notna())]
df = df[df["pricesold"].between(1000, 150000)]
df = df[df["Mileage"].between(0, 300000)]
df = df[df["Year"].between(1980, 2025)]
df.dropna(subset=["pricesold", "Year", "Model", "Make"], inplace=True)
df = df[~df["Model"].astype(str).str.strip().str.match(_BAD_MODEL_PATTERN)].copy()
df["is_bad_record"] = df.apply(_has_bad_text, axis=1)
df = df[~df["is_bad_record"]].copy()

# Feature engineering
df["brand_clean"] = df["Make"].apply(_normalize_brand)
df["brand_tier"] = df["brand_clean"].apply(_brand_tier)
df["zip3"] = df["zipcode"].astype(str).str[:3]
df["miles_per_year"] = df["Mileage"] / (df["Car_Age"] + 1)

for col in ["Engine", "Trim", "DriveType", "BodyType"]:
    df[col] = df[col].fillna("Missing").astype(str)

print(f"Dataset size after filtering: {len(df):,} rows")

# ══════════════════════════════════════════════════════════════════════════
#  TIME-AWARE SPLIT: Train on 2018-2019, Test on 2020
# ══════════════════════════════════════════════════════════════════════════
train_df = df[df["yearsold"].isin([2018, 2019])].copy()
test_df = df[df["yearsold"] == 2020].copy()

print(f"\nTrain set (2018-2019): {len(train_df):,} rows")
print(f"Test set  (2020):      {len(test_df):,} rows")
print(f"Split ratio:           {len(train_df)/(len(train_df)+len(test_df))*100:.1f}% / {len(test_df)/(len(train_df)+len(test_df))*100:.1f}%")

X_train = train_df[FEATURES]
y_train = train_df["pricesold"].values
X_test = test_df[FEATURES]
y_test = test_df["pricesold"].values

cat_indices = [FEATURES.index(c) for c in CAT_FEATURES]

# ── Train fresh models on 2018-2019 only ─────────────────────────────────
print("\nTraining Q25 model on 2018-2019 data...")
model_q25 = CatBoostRegressor(
    iterations=1500,
    depth=8,
    learning_rate=0.05,
    loss_function="Quantile:alpha=0.25",
    cat_features=cat_indices,
    verbose=100,
    random_seed=42,
)
model_q25.fit(X_train, y_train)

print("\nTraining Q50 model on 2018-2019 data...")
model_q50 = CatBoostRegressor(
    iterations=1500,
    depth=8,
    learning_rate=0.05,
    loss_function="Quantile:alpha=0.5",
    cat_features=cat_indices,
    verbose=100,
    random_seed=42,
)
model_q50.fit(X_train, y_train)

print("\nTraining Q75 model on 2018-2019 data...")
model_q75 = CatBoostRegressor(
    iterations=1500,
    depth=8,
    learning_rate=0.05,
    loss_function="Quantile:alpha=0.75",
    cat_features=cat_indices,
    verbose=100,
    random_seed=42,
)
model_q75.fit(X_train, y_train)

# ── Predict on BOTH sets ─────────────────────────────────────────────────
print("\nRunning predictions...")
train_pred_q25 = model_q25.predict(X_train)
train_pred_q50 = model_q50.predict(X_train)
train_pred_q75 = model_q75.predict(X_train)

test_pred_q25 = model_q25.predict(X_test)
test_pred_q50 = model_q50.predict(X_test)
test_pred_q75 = model_q75.predict(X_test)


def compute_metrics(y_true, pred_q25, pred_q50, pred_q75):
    """Compute all metrics for a given set of predictions."""
    mae = mean_absolute_error(y_true, pred_q50)
    median_ae = np.median(np.abs(y_true - pred_q50))
    rmse = np.sqrt(mean_squared_error(y_true, pred_q50))
    r2 = r2_score(y_true, pred_q50)
    mape = np.mean(np.abs((y_true - pred_q50) / y_true)) * 100
    within_20 = np.mean(np.abs(y_true - pred_q50) / y_true < 0.20) * 100
    within_30 = np.mean(np.abs(y_true - pred_q50) / y_true < 0.30) * 100

    # Quantile calibration
    below_q25 = np.mean(y_true < pred_q25) * 100
    below_q50 = np.mean(y_true < pred_q50) * 100
    below_q75 = np.mean(y_true < pred_q75) * 100
    in_iqr = np.mean((y_true >= pred_q25) & (y_true <= pred_q75)) * 100

    return {
        "mae": mae, "median_ae": median_ae, "rmse": rmse,
        "r2": r2, "mape": mape,
        "within_20": within_20, "within_30": within_30,
        "below_q25": below_q25, "below_q50": below_q50,
        "below_q75": below_q75, "in_iqr": in_iqr,
    }


train_m = compute_metrics(y_train, train_pred_q25, train_pred_q50, train_pred_q75)
test_m = compute_metrics(y_test, test_pred_q25, test_pred_q50, test_pred_q75)

# ══════════════════════════════════════════════════════════════════════════
#  1. SIDE-BY-SIDE: TRAIN vs TEST METRICS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  1. TRAIN vs TEST — REGRESSION METRICS (Q50 MODEL)")
print("=" * 70)

print(f"\n  {'Metric':<35} {'Train (2018-19)':>15} {'Test (2020)':>15}")
print(f"  {'-'*35} {'-'*15} {'-'*15}")
print(f"  {'MAE (Mean Absolute Error)':<35} {'${:,.0f}'.format(train_m['mae']):>15} {'${:,.0f}'.format(test_m['mae']):>15}")
print(f"  {'MedAE (Median Absolute Error)':<35} {'${:,.0f}'.format(train_m['median_ae']):>15} {'${:,.0f}'.format(test_m['median_ae']):>15}")
print(f"  {'RMSE':<35} {'${:,.0f}'.format(train_m['rmse']):>15} {'${:,.0f}'.format(test_m['rmse']):>15}")
print(f"  {'R² Score':<35} {train_m['r2']:>15.4f} {test_m['r2']:>15.4f}")
print(f"  {'MAPE (%)':<35} {train_m['mape']:>14.1f}% {test_m['mape']:>14.1f}%")
print(f"  {'% within 20%':<35} {train_m['within_20']:>14.1f}% {test_m['within_20']:>14.1f}%")
print(f"  {'% within 30%':<35} {train_m['within_30']:>14.1f}% {test_m['within_30']:>14.1f}%")

# ══════════════════════════════════════════════════════════════════════════
#  2. QUANTILE CALIBRATION — TRAIN vs TEST
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  2. QUANTILE CALIBRATION — TRAIN vs TEST")
print("=" * 70)

print(f"\n  {'Quantile':<30} {'Ideal':>8} {'Train':>8} {'Test':>8}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'% actual < Q25':<30} {'25.0%':>8} {train_m['below_q25']:>7.1f}% {test_m['below_q25']:>7.1f}%")
print(f"  {'% actual < Q50':<30} {'50.0%':>8} {train_m['below_q50']:>7.1f}% {test_m['below_q50']:>7.1f}%")
print(f"  {'% actual < Q75':<30} {'75.0%':>8} {train_m['below_q75']:>7.1f}% {test_m['below_q75']:>7.1f}%")
print(f"  {'% actual in [Q25, Q75]':<30} {'50.0%':>8} {train_m['in_iqr']:>7.1f}% {test_m['in_iqr']:>7.1f}%")

# ══════════════════════════════════════════════════════════════════════════
#  3. BASELINE COMPARISON ON TEST SET
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  3. BASELINE COMPARISON ON TEST SET (2020)")
print("=" * 70)

mean_pred = np.full_like(y_test, y_train.mean(), dtype=float)
median_pred = np.full_like(y_test, np.median(y_train), dtype=float)
np.random.seed(42)
shuffled = test_pred_q50.copy()
np.random.shuffle(shuffled)

mae_mean = mean_absolute_error(y_test, mean_pred)
r2_mean = r2_score(y_test, mean_pred)
mae_median = mean_absolute_error(y_test, median_pred)
r2_median = r2_score(y_test, median_pred)
mae_shuffled = mean_absolute_error(y_test, shuffled)
r2_shuffled = r2_score(y_test, shuffled)

print(f"\n  {'Method':<30} {'MAE':>10} {'R²':>8}")
print(f"  {'-'*30} {'-'*10} {'-'*8}")
print(f"  {'MODEL (Q50)':<30} {'${:,.0f}'.format(test_m['mae']):>10} {test_m['r2']:>8.4f}")
print(f"  {'Always predict train mean':<30} {'${:,.0f}'.format(mae_mean):>10} {r2_mean:>8.4f}")
print(f"  {'Always predict train median':<30} {'${:,.0f}'.format(mae_median):>10} {r2_median:>8.4f}")
print(f"  {'Shuffled predictions':<30} {'${:,.0f}'.format(mae_shuffled):>10} {r2_shuffled:>8.4f}")

improvement_mean = (1 - test_m['mae'] / mae_mean) * 100
improvement_median = (1 - test_m['mae'] / mae_median) * 100
print(f"\n  Model reduces MAE by {improvement_mean:.1f}% vs. mean baseline")
print(f"  Model reduces MAE by {improvement_median:.1f}% vs. median baseline")

# ══════════════════════════════════════════════════════════════════════════
#  4. PERFORMANCE BY PRICE SEGMENT (TEST SET)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  4. PERFORMANCE BY PRICE SEGMENT (TEST SET)")
print("=" * 70)

bins = [0, 3000, 5000, 10000, 20000, 40000, 80000, 150000]
labels = ["<$3k", "$3-5k", "$5-10k", "$10-20k", "$20-40k", "$40-80k", "$80k+"]
test_df = test_df.copy()
test_df["price_bin"] = pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)

print(f"\n  {'Segment':<12} {'Count':>8} {'MAE':>10} {'MAPE':>8} {'R²':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

for label in labels:
    mask = (test_df["price_bin"] == label).values
    if mask.sum() < 10:
        continue
    seg_y = y_test[mask]
    seg_pred = test_pred_q50[mask]
    seg_mae = mean_absolute_error(seg_y, seg_pred)
    seg_mape = np.mean(np.abs(seg_y - seg_pred) / seg_y) * 100
    seg_r2 = r2_score(seg_y, seg_pred) if len(seg_y) > 1 else float('nan')
    print(f"  {label:<12} {mask.sum():>8,} {'${:,.0f}'.format(seg_mae):>10} {seg_mape:>7.1f}% {seg_r2:>8.4f}")

# ══════════════════════════════════════════════════════════════════════════
#  5. PERFORMANCE BY BRAND TIER (TEST SET)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  5. PERFORMANCE BY BRAND TIER (TEST SET)")
print("=" * 70)

print(f"\n  {'Tier':<12} {'Count':>8} {'MAE':>10} {'MAPE':>8} {'R²':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

for tier in ["Economy", "Mid", "Luxury"]:
    mask = (test_df["brand_tier"] == tier).values
    if mask.sum() < 10:
        continue
    seg_y = y_test[mask]
    seg_pred = test_pred_q50[mask]
    seg_mae = mean_absolute_error(seg_y, seg_pred)
    seg_mape = np.mean(np.abs(seg_y - seg_pred) / seg_y) * 100
    seg_r2 = r2_score(seg_y, seg_pred)
    print(f"  {tier:<12} {mask.sum():>8,} {'${:,.0f}'.format(seg_mae):>10} {seg_mape:>7.1f}% {seg_r2:>8.4f}")

# ══════════════════════════════════════════════════════════════════════════
#  6. GENERATE DIAGNOSTIC PLOTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  6. GENERATING DIAGNOSTIC PLOTS → eval_temporal_plots.png")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Time-Aware Validation: Train on 2018-2019, Test on 2020",
             fontsize=14, fontweight='bold', y=1.01)

# Plot 1: Actual vs Predicted on TEST set
ax = axes[0, 0]
sample_idx = np.random.RandomState(42).choice(len(y_test), size=min(5000, len(y_test)), replace=False)
ax.scatter(y_test[sample_idx], test_pred_q50[sample_idx], alpha=0.15, s=8, c='steelblue')
lims = [0, np.percentile(y_test, 99)]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
ax.set_xlabel('Actual Price ($)')
ax.set_ylabel('Predicted Price ($)')
ax.set_title(f'Actual vs Predicted — TEST SET\nR²={test_m["r2"]:.3f}, MAE=${test_m["mae"]:,.0f}')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.legend()

# Plot 2: Residual distribution on TEST set
ax = axes[0, 1]
test_residuals = y_test - test_pred_q50
clipped_res = np.clip(test_residuals, np.percentile(test_residuals, 1), np.percentile(test_residuals, 99))
ax.hist(clipped_res, bins=100, color='steelblue', edgecolor='none', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Residual (Actual - Predicted) $')
ax.set_ylabel('Count')
ax.set_title('Residual Distribution — TEST SET')

# Plot 3: Train vs Test metric comparison
ax = axes[0, 2]
metrics_compare = ['R²', 'MAE\n($K)', 'MAPE\n(%)', 'Within\n20%']
train_vals = [train_m['r2'], train_m['mae']/1000, train_m['mape'], train_m['within_20']]
test_vals = [test_m['r2'], test_m['mae']/1000, test_m['mape'], test_m['within_20']]
x_pos = np.arange(len(metrics_compare))
width = 0.35
bars1 = ax.bar(x_pos - width/2, train_vals, width, label='Train (2018-19)', color='steelblue', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, test_vals, width, label='Test (2020)', color='coral', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics_compare)
ax.set_title('Train vs Test Metrics')
ax.legend()
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

# Plot 4: MAPE by price segment (TEST set)
ax = axes[1, 0]
seg_data = []
for label in labels:
    mask = (test_df["price_bin"] == label).values
    if mask.sum() >= 10:
        seg_mape = np.mean(np.abs(y_test[mask] - test_pred_q50[mask]) / y_test[mask]) * 100
        seg_data.append((label, seg_mape, mask.sum()))

seg_labels = [s[0] for s in seg_data]
seg_mapes = [s[1] for s in seg_data]
seg_counts = [s[2] for s in seg_data]
bars = ax.bar(seg_labels, seg_mapes, color='steelblue', alpha=0.7)
ax.set_xlabel('Price Segment')
ax.set_ylabel('MAPE (%)')
ax.set_title('MAPE by Price Segment — TEST SET')
ax.tick_params(axis='x', rotation=45)
for bar, count in zip(bars, seg_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'n={count:,}', ha='center', va='bottom', fontsize=8)

# Plot 5: Quantile calibration on TEST set
ax = axes[1, 1]
quantiles_labels = ['Q25', 'Q50', 'Q75']
actual_coverage = [test_m['below_q25'], test_m['below_q50'], test_m['below_q75']]
ideal_coverage = [25.0, 50.0, 75.0]
x_pos = np.arange(len(quantiles_labels))
width = 0.35
ax.bar(x_pos - width/2, actual_coverage, width, label='Actual (Test)', color='steelblue', alpha=0.7)
ax.bar(x_pos + width/2, ideal_coverage, width, label='Ideal', color='coral', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(quantiles_labels)
ax.set_ylabel('% of actuals below quantile')
ax.set_title('Quantile Calibration — TEST SET')
ax.legend()

# Plot 6: Model vs baselines (TEST set)
ax = axes[1, 2]
methods = ['Model\n(Q50)', 'Train Mean\nBaseline', 'Train Median\nBaseline', 'Shuffled\n(Random)']
r2_values = [test_m['r2'], r2_mean, r2_median, r2_shuffled]
colors = ['forestgreen', 'gray', 'gray', 'indianred']
bars = ax.bar(methods, r2_values, color=colors, alpha=0.8)
ax.set_ylabel('R² Score')
ax.set_title('R² — Model vs Baselines (TEST SET)')
ax.axhline(0, color='black', linewidth=0.5)
for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(_DIR / "eval_temporal_plots.png", dpi=150, bbox_inches='tight')
print("  Saved to eval_temporal_plots.png")

# ══════════════════════════════════════════════════════════════════════════
#  FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FINAL VERDICT — TIME-AWARE VALIDATION")
print("=" * 70)

r2_drop = train_m['r2'] - test_m['r2']
mae_increase = (test_m['mae'] - train_m['mae']) / train_m['mae'] * 100

print(f"""
  TRAIN (2018-2019) → TEST (2020) performance:

  R²:   {train_m['r2']:.4f} → {test_m['r2']:.4f}  (drop: {r2_drop:.4f})
  MAE:  ${train_m['mae']:,.0f} → ${test_m['mae']:,.0f}  (increase: {mae_increase:+.1f}%)
  MAPE: {train_m['mape']:.1f}% → {test_m['mape']:.1f}%

  Quantile calibration on unseen 2020 data:
    Q25: {test_m['below_q25']:.1f}% (ideal 25%)
    Q50: {test_m['below_q50']:.1f}% (ideal 50%)
    Q75: {test_m['below_q75']:.1f}% (ideal 75%)

  vs. baselines on test set:
    {improvement_mean:.1f}% better MAE than always predicting the training mean
    {improvement_median:.1f}% better MAE than always predicting the training median

  This is an HONEST out-of-sample evaluation. The model was trained on
  2018-2019 data and tested on 2020 data it has never seen — simulating
  real-world deployment where the model predicts future prices from
  historical patterns.
""")
