# CarPrice

CarPrice is a Streamlit app for used-car decision support. It is built around two user roles:

- Sellers who want a data-driven estimate of what their car is worth today
- Buyers who want to find used cars in the dataset that look like strong value relative to a pricing model

The app combines a quantile regression pricing model, buyer-side ranking logic, depreciation analysis, and regional market comparisons into a single interactive workflow.

## What The App Does

### Seller Experience

For a seller, the app can:

- estimate a competitive, fair-market, and premium price range
- show where the car sits on a depreciation curve
- estimate how much value may be lost if the seller waits longer to sell
- simulate how price changes if the car is driven more miles or sold a few years later
- provide pricing tips based on the estimate and depreciation stage

### Buyer Experience

For a buyer, the app can:

- filter cars by budget, body type, drive type, make, mileage, and minimum year
- rank individual vehicle records by comparing observed price against the model's fair-value estimate
- highlight the top deals within the buyer's budget range
- show image previews when available
- compare shortlisted cars side by side
- summarize which US regions tend to show the best deals overall

## Current Modeling Logic

### Price Prediction

The seller pricing module uses three trained CatBoost quantile models:

- `model_q25.cbm`
- `model_q50.cbm`
- `model_q75.cbm`

These are used in `predict.py` to produce:

- competitive price
- fair-market estimate
- premium price

The current prediction logic also:

- aligns vehicle age to the original training era
- applies a CPI-style adjustment so estimates are closer to current-dollar values

### Buyer Recommendation

The buyer recommendation module is implemented in `buyer_recommend.py`.

It works by:

1. filtering historical sold-car records using the buyer's constraints
2. predicting a fair value for each candidate using the pricing model
3. comparing actual observed price against predicted fair value
4. ranking cars by a combination of value gap and budget fit

The current buyer view recommends individual vehicle records rather than aggregated model families.

### Depreciation And Regional Analysis

The seller and buyer dashboards also use `depreciation.py` to:

- compute average price by vehicle age
- estimate future depreciation
- evaluate regional pricing residuals

Regional analysis now focuses on broad US regions and excludes non-US / unknown zip-based regions from the final display.

## Main Files

- `streamlit_app.py`
  Main application entry point and UI flow

- `predict.py`
  Seller-side quantile price prediction

- `buyer_recommend.py`
  Buyer-side filtering, scoring, and recommendation logic

- `depreciation.py`
  Depreciation calculations and regional deal analysis

- `car_image.py`
  Vehicle image lookup helper using Wikipedia search thumbnails

- `api.py`
  Optional FastAPI wrapper for seller and buyer endpoints

- `used_car_sales.csv`
  Main dataset used by the app

- `model_q25.cbm`, `model_q50.cbm`, `model_q75.cbm`
  Trained CatBoost pricing models

## Research / Evaluation Files

These files support model development and evaluation and are not required to launch the Streamlit app:

- `benchmark_quantile_models.py`
- `evaluate_temporal.py`
- `plot_quantile_benchmark.py`
- `quantile_benchmark_cv_results.csv`
- `quantile_benchmark_cv_results_folds.csv`
- `quantile_benchmark_plots.png`
- `eval_temporal_plots.png`

## How To Run

Install the required Python packages in the same environment:

```bash
pip install streamlit catboost pandas numpy plotly fastapi uvicorn pydantic
```

Then start the Streamlit app from the repository root:

```bash
streamlit run streamlit_app.py
```

You can still run the app locally this way without Docker if you already have the Python environment set up.

## Run With Docker

If you want a portable environment that runs the app the same way on different machines, you can use Docker instead.

Build the image from the repository root:

```bash
docker build -t carprice .
```

Then start a container:

```bash
docker run -p 8501:8501 carprice
```

After that, open:

```txt
http://localhost:8501
```

Notes:

- `docker build -t carprice .` builds a Docker image named `carprice` from the current folder and its `Dockerfile`
- `docker run -p 8501:8501 carprice` starts a container from that image and maps your local port `8501` to the app inside the container
- after code changes, rebuild the image before rerunning the container

## Optional API Mode

If you want to run the FastAPI wrapper instead of the Streamlit app:

```bash
python -m uvicorn api:app --reload
```

This exposes:

- `GET /health`
- `POST /api/seller/price`
- `POST /api/buyer/recommend`

## Notes

- Buyer recommendations are based on historical sale records in the dataset, not live marketplace inventory.
- Vehicle images are fetched from Wikipedia thumbnails when available, so some cards may not have images.
- The app currently uses the historical dataset included in the repository and does not yet connect to external listing sources.

## Project Status

This repository reflects the completed current version of the CarPrice app:

- Streamlit-based UI
- seller pricing workflow
- buyer recommendation workflow
- depreciation analysis
- regional best-deal insights
- model benchmarking and evaluation scripts
