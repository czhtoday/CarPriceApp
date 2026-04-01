from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    data_path = APP_DIR / "used_car_sales.csv"
    df = pd.read_csv(data_path)

    for col in ["pricesold", "Mileage", "Year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_resource
def load_prediction_module():
    import predict
    return predict


@st.cache_resource
def load_recommendation_module():
    import buyer_recommend
    return buyer_recommend


@st.cache_resource
def load_image_module():
    import car_image
    return car_image


@st.cache_resource
def load_depreciation_module():
    import depreciation
    return depreciation


def option_list(df: pd.DataFrame, column: str) -> list[str]:
    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    values = values[values.ne("") & values.ne("nan")]
    return sorted(values.unique().tolist())


def format_currency(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return "${:,.0f}".format(value)


@st.cache_data(show_spinner=False)
def fetch_vehicle_image(title: str, typical_year: int) -> bytes | None:
    car_image = load_image_module()
    image_url = car_image.get_vehicle_image(title, typical_year)
    if not image_url:
        return None
    return car_image.fetch_image_bytes(image_url)


def predict_mid_from_row(row: pd.Series) -> float | None:
    """
    Wrapper used by depreciation.py region analysis.
    Returns the fair market price (mid) for one dataset row.
    """
    try:
        predict = load_prediction_module()
        _, mid, _ = predict.get_price_range(
            make=row.get("Make", "Missing"),
            model=row.get("Model", "Missing"),
            year=int(row.get("Year", 2019)),
            mileage=int(row.get("Mileage", 0)),
            body_type=row.get("BodyType", "Missing"),
            drive_type=row.get("DriveType", "Missing"),
            zipcode=row.get("zipcode", "Missing"),
            engine=row.get("Engine", "Missing"),
            trim=row.get("Trim", "Missing"),
        )
        return mid
    except Exception:
        return None


def render_seller_tab(df: pd.DataFrame) -> None:
    st.subheader("Seller Pricing")
    st.write("Estimate a competitive, fair, and premium asking price for a used car.")

    makes = option_list(df, "Make")
    body_types = option_list(df, "BodyType")
    drive_types = option_list(df, "DriveType")

    with st.form("seller_form"):
        col1, col2 = st.columns(2)
        with col1:
            make = st.selectbox("Make", makes, index=makes.index("Toyota") if "Toyota" in makes else 0)
            model = st.text_input("Model", value="RAV4")
            year = st.number_input("Year", min_value=1980, max_value=2026, value=2018, step=1)
            mileage = st.number_input("Mileage", min_value=0, max_value=400000, value=45000, step=1000)
            zipcode = st.text_input("Zip code", value="90210")
        with col2:
            body_type = st.selectbox("Body type", body_types, index=body_types.index("SUV") if "SUV" in body_types else 0)
            drive_type = st.selectbox("Drive type", drive_types, index=drive_types.index("AWD") if "AWD" in drive_types else 0)
            trim = st.text_input("Trim", value="XLE")
            engine = st.text_input("Engine", value="2.5L I4")

        submitted = st.form_submit_button("Estimate price range", use_container_width=True)

    if not submitted:
        return

    try:
        predict = load_prediction_module()
        low, mid, high = predict.get_price_range(
            make=make,
            model=model,
            year=int(year),
            mileage=int(mileage),
            body_type=body_type,
            drive_type=drive_type,
            zipcode=zipcode,
            engine=engine,
            trim=trim,
        )
    except Exception as exc:
        st.error("Unable to run the pricing model right now: {}".format(exc))
        return

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Competitive", format_currency(low))
    metric2.metric("Fair market", format_currency(mid))
    metric3.metric("Premium", format_currency(high))

    st.caption(
        "Competitive helps the car sell faster, fair is the midpoint estimate, and premium is a more ambitious asking price."
    )

    # =========================
    # Depreciation insight
    # =========================
    try:
        depreciation = load_depreciation_module()
        dep_df = depreciation.load_data()
        avg_price = depreciation.compute_depreciation(dep_df)

        car_age = max(2019 - int(year), 0)
        future_drop = depreciation.estimate_future_drop(avg_price, car_age, years=2)

        st.markdown("### Depreciation Insight")
        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:
            st.metric("Estimated value drop in next 2 years", format_currency(future_drop))

        with insight_col2:
            if car_age <= 8:
                stage = "High depreciation stage"
            elif car_age <= 15:
                stage = "Moderate depreciation stage"
            else:
                stage = "Low depreciation stage"
            st.metric("Current depreciation stage", stage)

        st.info(
            "This estimate is based on the average depreciation pattern by car age in the dataset."
        )
    except Exception as exc:
        st.warning("Depreciation insight unavailable right now: {}".format(exc))


def render_buyer_tab(df: pd.DataFrame) -> None:
    st.subheader("Buyer Recommendation")
    st.write("Filter by budget and preferences, then rank cars by estimated value.")

    body_types = option_list(df, "BodyType")
    drive_types = ["Any"] + option_list(df, "DriveType")
    makes = ["Any"] + option_list(df, "Make")

    with st.form("buyer_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            budget = st.number_input("Budget", min_value=1000, max_value=150000, value=20000, step=500)
            body_type = st.selectbox("Body type", body_types, index=body_types.index("SUV") if "SUV" in body_types else 0)
        with col2:
            drive_type = st.selectbox("Drive type (optional)", drive_types, index=0)
            use_max_mileage = st.checkbox("Max mileage filter (optional)", value=False)
            max_mileage = st.number_input(
                "Max mileage",
                min_value=0,
                max_value=300000,
                value=80000,
                step=5000,
                disabled=not use_max_mileage,
            )
        with col3:
            make = st.selectbox("Preferred make (optional)", makes, index=0)
            use_min_year = st.checkbox("Min year filter (optional)", value=False)
            min_year = st.number_input(
                "Min year",
                min_value=1980,
                max_value=2026,
                value=2015,
                step=1,
                disabled=not use_min_year,
            )

        zipcode = st.text_input("Zip code (optional)", value="")
        submitted = st.form_submit_button("Find recommendations", use_container_width=True)

    if not submitted:
        return

    try:
        buyer_recommend = load_recommendation_module()
        results = buyer_recommend.recommend(
            budget=float(budget),
            body_type=body_type,
            drive_type=None if drive_type == "Any" else drive_type,
            make=None if make == "Any" else make,
            max_mileage=int(max_mileage) if use_max_mileage and max_mileage else None,
            min_year=int(min_year) if use_min_year and min_year else None,
            zipcode=zipcode or None,
            top_n=10,
        )
    except Exception as exc:
        st.error("Unable to run the recommendation model right now: {}".format(exc))
        return

    if not results:
        st.warning("No matches found for these filters. Try loosening the mileage, year, or make constraints.")
        return

    top_pick = results[0]
    st.success(
        "Top pick: {title} ({year_range}), typical price {price}, estimated fair value {fair}, confidence {confidence}.".format(
            title=top_pick["title"],
            year_range=top_pick["year_range"],
            price=format_currency(top_pick["typical_price"]),
            fair=format_currency(top_pick["predicted_fair"]),
            confidence=top_pick["confidence"],
        )
    )

    for start in range(0, len(results), 2):
        cols = st.columns(2)
        for col, rec in zip(cols, results[start : start + 2]):
            with col:
                with st.container(border=True):
                    image_url = None
                    if start < 4:
                        image_url = fetch_vehicle_image(rec["title"], rec["typical_year"])
                    if image_url:
                        st.image(image_url, use_container_width=True)
                    else:
                        st.info("Image unavailable")

                    title_col, badge_col = st.columns([4, 1])
                    with title_col:
                        st.markdown("#### {}".format(rec["title"]))
                    with badge_col:
                        st.markdown("**{}**".format(rec["confidence"]))

                    st.write("Typical year: {}".format(rec["typical_year"]))
                    st.write("Typical mileage: {:,} miles".format(rec["typical_mileage"]))
                    st.write("Typical price: {}".format(format_currency(rec["typical_price"])))
                    st.write("Estimated fair value: {}".format(format_currency(rec["predicted_fair"])))

    # =========================
    # Region deals insight
    # =========================
    try:
        depreciation = load_depreciation_module()
        top_regions = depreciation.get_region_deals(predict_mid_from_row)

        st.markdown("### Best Regions to Find Better Deals")
        st.caption(
            "Price advantage shows how much cheaper cars are in a region compared with model expectations."
        )

        best_region = top_regions.iloc[0]
        st.success(
            "Best current region: {region} — buyers save about {save} on average ({label}).".format(
                region=best_region["region"],
                save=format_currency(best_region["price_advantage"]),
                label=best_region["deal_label"],
            )
        )

        display_regions = top_regions[["region", "price_advantage", "deal_label", "sample_size"]].copy()
        display_regions.columns = ["Region", "You Save", "Deal Level", "Sample Size"]
        display_regions["You Save"] = display_regions["You Save"].apply(format_currency)

        st.dataframe(display_regions, use_container_width=True)

        chart_df = top_regions.copy()
        chart_df = chart_df.sort_values("price_advantage", ascending=True)

        st.bar_chart(
            data=chart_df.set_index("region")["price_advantage"],
            use_container_width=True,
        )

    except Exception as exc:
        st.error("Regional insight error: {}".format(exc))


def main() -> None:
    st.set_page_config(
        page_title="CarPrice Demo",
        page_icon="🚗",
        layout="wide",
    )

    st.title("CarPrice Streamlit Demo")
    st.caption("Direction 1: seller pricing. Direction 2: buyer recommendation. Direction 3: depreciation and region insights.")

    try:
        df = load_reference_data()
    except Exception as exc:
        st.error("Unable to load the project dataset: {}".format(exc))
        return

    tab1, tab2 = st.tabs(["Seller", "Buyer"])
    with tab1:
        render_seller_tab(df)
    with tab2:
        render_buyer_tab(df)


if __name__ == "__main__":
    main()