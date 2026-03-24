# CarPrice

CarPrice is a used-car decision support app built for two types of users:

- Sellers who want a reasonable asking price for their car
- Buyers who want to find used cars that fit their budget and offer strong value

This repository currently contains the implemented work for:

1. Seller pricing
2. Buyer recommendation

Later project directions can be added to this repository as the team continues development.

## Project Motivation

Buying or selling a used car often involves a lot of uncertainty. Sellers may not know how much their car is worth, and buyers may struggle to identify which vehicles are actually good value within their budget.

Our project aims to turn historical used-car sales data into a simple web experience that gives users practical guidance based on their role:

- Sellers receive a suggested price range for their vehicle
- Buyers receive filtered and ranked recommendations based on preferences and budget

## Current Features

### Seller Pricing

The seller-facing module estimates a price range for a vehicle using its:

- make
- model
- year
- mileage
- body type
- drive type
- zip code
- optional trim and engine details

The pricing output includes:

- a competitive price
- a fair market estimate
- a premium price

This module is powered by three trained CatBoost quantile models and is exposed through `predict.py`.

### Buyer Recommendation

The buyer-facing module helps users search for cars that match their constraints, such as:

- budget
- body type
- drive type
- preferred make
- maximum mileage
- minimum year

Candidate vehicles are ranked by comparing their historical sale price against the fair-value estimate produced by the seller pricing model.

This means the recommendation system builds directly on top of the pricing model instead of acting as a completely separate pipeline.

This module is implemented in `buyer_recommend.py`.

### Streamlit Demo

The current interface is a Streamlit app with two views:

- `Seller`: enter a car's details and estimate a price range
- `Buyer`: enter budget and preferences and view ranked recommendations

The demo entry point is `streamlit_app.py`.

## Repository Structure

- `streamlit_app.py`: Streamlit interface for both user roles
- `predict.py`: seller pricing inference logic
- `buyer_recommend.py`: buyer recommendation logic
- `used_car_sales.csv`: dataset used by the current demo
- `model_q25.cbm`: 25th percentile pricing model
- `model_q50.cbm`: 50th percentile pricing model
- `model_q75.cbm`: 75th percentile pricing model
- `feature_medians.json`: saved numeric feature statistics

## How to Run

Install the required dependencies in the same Python environment:

```bash
pip install streamlit catboost pandas numpy
```

Then launch the app from the repository root:

```bash
streamlit run streamlit_app.py
```

## Current Scope

This repository reflects the currently implemented part of the project:

- seller pricing
- buyer recommendation

The project is still evolving, and additional modules can be added by other team members as new directions are completed.
