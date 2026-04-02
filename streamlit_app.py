from __future__ import annotations
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


@st.cache_resource
def load_predict_module():
    import predict
    return predict


@st.cache_resource
def load_recommend_module():
    import buyer_recommend
    return buyer_recommend


@st.cache_resource
def load_depreciation_module():
    import depreciation
    return depreciation


@st.cache_resource
def load_car_image_module():
    import car_image
    return car_image


def get_cached_image(title: str, typical_year: int | None) -> bytes | None:
    cache = st.session_state.setdefault("_img_cache", {})
    key = (title, typical_year)
    if key in cache:
        return cache[key]
    car_image = load_car_image_module()
    url = car_image.get_vehicle_image(title, typical_year)
    if url:
        data = car_image.fetch_image_bytes(url)
        if data:
            cache[key] = data
            return data
    return None


@st.cache_data(show_spinner="Analyzing regions…")
def get_cached_region_deals(top_n: int) -> pd.DataFrame:
    depreciation_mod = load_depreciation_module()
    df = load_reference_data().sample(n=3000, random_state=42).copy()
    df["zip3"] = df["zipcode"].fillna("Missing").astype(str).str[:3]
    region_analysis = depreciation_mod.compute_region_analysis(df, row_predict_func)
    return region_analysis.head(top_n)


@st.cache_data(show_spinner="Finding best matches…")
def get_cached_recommend(
    budget: float,
    body_type: str | None,
    drive_type: str | None,
    make: str | None,
    max_mileage: int | None,
    min_year: int | None,
    zipcode: str | None,
    top_n: int,
) -> list:
    recommend_mod = load_recommend_module()
    return recommend_mod.recommend(
        budget=budget, body_type=body_type, drive_type=drive_type,
        make=make, max_mileage=max_mileage, min_year=min_year,
        zipcode=zipcode, top_n=top_n,
    )


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    data_path = APP_DIR / "used_car_sales.csv"
    df = pd.read_csv(data_path)
    for col in ["pricesold", "Mileage", "Year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fmt(value):
    return "${:,.0f}".format(value)


def option_list(df, column, normalizer=None, min_count=0):
    values = df[column].dropna().astype(str).str.strip()
    values = values[values.ne("") & values.ne("nan")]
    if normalizer is not None:
        values = values.map(normalizer)
        values = values[~values.isin(("Other", "Missing"))]
    if min_count > 1:
        counts = values.value_counts()
        values = values[values.isin(counts[counts >= min_count].index)]
    return sorted(values.unique().tolist())


def preferred_body_type_options(body_types):
    preferred_order = ["SUV", "Sedan", "Pickup", "Coupe", "Convertible"]
    return [body for body in preferred_order if body in body_types]


def nav_to(page):
    st.session_state["page"] = page


def row_predict_func(row):
    predict = load_predict_module()
    try:
        _, mid, _ = predict.get_price_range(
            make=str(row.get("Make", "Missing")),
            model=str(row.get("Model", "Missing")),
            year=int(row.get("Year", 2015)),
            mileage=int(row.get("Mileage", 50000)),
            body_type=str(row.get("BodyType", "Missing")),
            drive_type=str(row.get("DriveType", "Missing")),
            zipcode=str(row.get("zipcode", "Missing")),
            engine=str(row.get("Engine", "Missing")),
            trim=str(row.get("Trim", "Missing")),
        )
        return mid
    except Exception:
        return None


def page_role():
    st.markdown(
        "<div style='text-align:center; padding-top:30px;'>"
        "<div style='font-size:48px;'>🚗</div>"
        "<h1 style='margin-bottom:0;'>CarPrice</h1>"
        "<p style='color:gray; font-size:16px; margin-bottom:40px;'>"
        "A used-car decision assistant for pricing, value comparison, and market timing.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "Choose a path to get started. Sellers receive a pricing recommendation and resale outlook. "
        "Buyers see the strongest-value cars in the dataset for their budget."
    )

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### 💰 I'm a Seller")
            st.write(
                "Estimate what your car is worth today and see whether selling now or later is the better move."
            )
            st.markdown(
                "- Recommended asking-price range\n"
                "- Depreciation stage analysis\n"
                "- Hold-or-sell simulator"
            )
            if st.button("Get Started as Seller", use_container_width=True, type="primary"):
                st.session_state["role"] = "seller"
                nav_to("profile")
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("### 🔍 I'm a Buyer")
            st.write(
                "Find cars that look like strong value for your budget by comparing listed prices with model-based fair value."
            )
            st.markdown(
                "- Top 10 value-ranked recommendations\n"
                "- Estimated savings vs fair value\n"
                "- Data-support labels\n"
                "- Regional best-deal insights"
            )
            if st.button("Get Started as Buyer", use_container_width=True, type="primary"):
                st.session_state["role"] = "buyer"
                nav_to("profile")
                st.rerun()


def page_profile():
    role = st.session_state.get("role", "seller")
    is_seller = role == "seller"

    if st.button("← Back to role selection"):
        nav_to("role")
        st.rerun()

    df = load_reference_data()
    predict = load_predict_module()
    recommend = load_recommend_module()
    makes = option_list(df, "Make", normalizer=predict._normalize_brand, min_count=10)
    body_types = preferred_body_type_options(
        option_list(df, "BodyType", normalizer=recommend._normalize_body_type)
    )
    drive_types = option_list(df, "DriveType", normalizer=recommend._normalize_drive_type)

    st.title("💰 Tell us about your car" if is_seller else "🔍 Set your preferences")
    st.caption(
        "Enter your vehicle details to get a price recommendation and resale outlook."
        if is_seller
        else "Set your budget and filters to surface the most compelling deals in the dataset."
    )

    if is_seller:
        with st.form("seller_form"):
            c1, c2 = st.columns(2)
            with c1:
                make = st.selectbox("Make", makes, index=makes.index("Toyota") if "Toyota" in makes else 0)
                model = st.text_input("Model", value="RAV4")
                year = st.number_input("Year", min_value=1990, max_value=2026, value=2018)
                mileage = st.number_input("Mileage", min_value=0, max_value=400000, value=45000, step=1000)
                zipcode = st.text_input("Zip Code", value="90210", help="For regional pricing")
            with c2:
                body = st.selectbox("Body Type", body_types, index=body_types.index("SUV") if "SUV" in body_types else 0)
                drive = st.selectbox("Drive Type", drive_types, index=drive_types.index("AWD") if "AWD" in drive_types else 0)
                trim = st.text_input("Trim (optional)", placeholder="e.g. XLE")
                engine = st.text_input("Engine (optional)", placeholder="e.g. 2.5L I4")

            if st.form_submit_button("Get My Estimate →", use_container_width=True, type="primary"):
                st.session_state["profile"] = dict(
                    make=make, model=model, year=int(year), mileage=int(mileage),
                    body=body, drive=drive, zip=zipcode, trim=trim, engine=engine,
                )
                nav_to("seller_dash")
                st.rerun()
    else:
        with st.form("buyer_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                budget = st.number_input("Budget ($)", min_value=1000, max_value=150000, value=18000, step=500)
                body = st.selectbox("Body Type", ["Any"] + body_types, index=0)
            with c2:
                drive = st.selectbox("Drive Type (optional)", ["Any"] + drive_types, index=0)
                use_max_mil = st.checkbox("Max mileage filter", value=False)
                max_mil = st.number_input("Max mileage", min_value=0, max_value=300000, value=80000, step=5000)
            with c3:
                make = st.selectbox("Preferred Make", ["Any"] + makes, index=0)
                use_min_yr = st.checkbox("Min year filter", value=False)
                min_yr = st.number_input("Min year", min_value=1990, max_value=2026, value=2015)
            zipcode = st.text_input("Zip Code (optional)", value="")

            if st.form_submit_button("Find Recommendations →", use_container_width=True, type="primary"):
                st.session_state["profile"] = dict(
                    budget=float(budget),
                    body=body if body != "Any" else None,
                    drive=drive if drive != "Any" else None,
                    make=make if make != "Any" else None,
                    max_mil=int(max_mil) if use_max_mil else None,
                    min_yr=int(min_yr) if use_min_yr else None,
                    zip=zipcode or None,
                )
                nav_to("buyer_dash")
                st.rerun()


def page_seller_dash():
    p = st.session_state.get("profile")
    if not p:
        nav_to("role"); st.rerun(); return

    if st.button("← Start over"):
        nav_to("role"); st.rerun()

    predict = load_predict_module()
    depreciation = load_depreciation_module()

    try:
        low, mid, high = predict.get_price_range(
            make=p["make"], model=p["model"], year=p["year"],
            mileage=p["mileage"], body_type=p["body"], drive_type=p["drive"],
            zipcode=p.get("zip", ""), engine=p.get("engine", ""),
            trim=p.get("trim", ""),
        )
    except Exception as e:
        st.error(f"Pricing model error: {e}")
        return

    car_age = max(2019 - p["year"], 0)

    st.title(f"🚗 {p['year']} {p['make']} {p['model']}")
    tier = "Luxury" if p["make"] in ["BMW","Mercedes-Benz","Audi","Lexus","Porsche","Tesla","Volvo","Cadillac","Jaguar","Land Rover","Acura","Infiniti","Lincoln","Maserati"] else "Mid" if p["make"] in ["Toyota","Honda","Ford","Chevrolet","Hyundai","Kia","Subaru","Mazda","Nissan","Jeep","Dodge","Ram","GMC","Volkswagen","Chrysler","Buick","Mini"] else "Economy"
    st.caption(f"{p['mileage']:,} miles · {p['body']} · {p['drive']}"
               + (f" · {p['trim']}" if p.get("trim") else "")
               + f" · {tier} tier")

    st.divider()

    st.subheader("📊 Price Estimate")
    st.write("These three price points translate the model output into practical selling strategies.")

    m1, m2, m3 = st.columns(3)
    m1.metric("⚡ Sell-Fast Price", fmt(low), help="A more competitive number if you want stronger short-term interest")
    m2.metric("⚖️ Recommended Price", fmt(mid), help="The main fair-market estimate for a balanced listing")
    m3.metric("💎 Premium Price", fmt(high), help="A more ambitious price that may take longer to sell")

    fig_g = go.Figure(go.Indicator(
        mode="gauge+number", value=mid,
        number={"prefix": "$", "font": {"size": 28}},
        title={"text": "Fair Market Value"},
        gauge={
            "axis": {"range": [low * 0.7, high * 1.2], "tickprefix": "$"},
            "bar": {"color": "#dea03a"},
            "steps": [
                {"range": [low * 0.7, low], "color": "rgba(47,200,114,0.15)"},
                {"range": [low, high], "color": "rgba(222,160,58,0.12)"},
                {"range": [high, high * 1.2], "color": "rgba(232,80,80,0.12)"},
            ],
        },
    ))
    fig_g.update_layout(height=250, margin=dict(t=50, b=20, l=40, r=40))
    st.plotly_chart(fig_g, use_container_width=True)

    st.divider()

    st.subheader("📉 Depreciation Analysis")
    st.write("See where your car sits on the value curve and what delaying the sale might cost.")

    try:
        dep_df = depreciation.load_data()
        avg_price = depreciation.compute_depreciation(dep_df)

        if not avg_price.empty:
            if car_age <= 2:
                stage, stage_color = "Steep Drop", "red"
                stage_desc = "Your car is in the steepest depreciation phase. New cars typically lose 15–25% in the first 2 years."
            elif car_age <= 5:
                stage, stage_color = "Moderate Decline", "orange"
                stage_desc = "Depreciation is slowing but still significant. Often a good time to sell before the curve flattens."
            elif car_age <= 10:
                stage, stage_color = "Gradual Plateau", "blue"
                stage_desc = "Depreciation has slowed considerably. Value is relatively stable at this stage."
            else:
                stage, stage_color = "Stable Floor", "green"
                stage_desc = "Most depreciation has occurred. Remaining value holds steady."

            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"**Current Stage:** :{stage_color}[{stage}]")
                st.write(stage_desc)
                st.caption(f"Vehicle age: {car_age} years")
            with d2:
                future_drop = depreciation.estimate_future_drop(avg_price, car_age, years=2)
                st.metric(
                    "Estimated loss if you keep 2 more years",
                    f"-{fmt(abs(future_drop))}" if future_drop else "N/A",
                    delta=f"~{fmt(abs(future_drop))} less" if future_drop else None,
                    delta_color="inverse",
                )

            fig_dep = go.Figure()
            fig_dep.add_trace(go.Scatter(
                x=avg_price["Car_Age"], y=avg_price["pricesold"],
                mode="lines+markers", name="Avg Sale Price",
                line=dict(color="#dea03a", width=3),
                marker=dict(size=5), fill="tozeroy",
                fillcolor="rgba(222,160,58,0.08)",
            ))
            current_row = avg_price[avg_price["Car_Age"] == car_age]
            if not current_row.empty:
                cv = current_row.iloc[0]["pricesold"]
                fig_dep.add_trace(go.Scatter(
                    x=[car_age], y=[cv], mode="markers+text",
                    marker=dict(size=14, color="#e85050", symbol="diamond"),
                    text=[f"Your car: {fmt(cv)}"], textposition="top center",
                    textfont=dict(size=11, color="#e85050"), name="Your Car",
                ))
            fig_dep.update_layout(
                title="Average Sale Price by Vehicle Age (all makes)",
                xaxis_title="Vehicle Age (years)", yaxis_title="Avg Sale Price ($)",
                height=350, margin=dict(t=50, b=40), yaxis=dict(tickprefix="$"),
            )
            st.plotly_chart(fig_dep, use_container_width=True)
        else:
            st.warning("Not enough data to compute depreciation curve.")
    except Exception as e:
        st.warning(f"Depreciation analysis unavailable: {e}")

    st.divider()

    st.subheader("🔮 Hold-or-Sell Simulator")
    st.caption("See how your estimate changes if you keep the car longer and add more miles before selling.")

    with st.form("whatif"):
        w1, w2 = st.columns(2)
        with w1:
            years_to_wait = st.selectbox(
                "When would you sell?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: "Sell now" if x == 0 else "In {} year{}".format(x, "" if x == 1 else "s"),
                index=0,
            )
        with w2:
            extra_miles = st.number_input(
                "Additional miles before selling",
                min_value=0,
                max_value=200000,
                value=0,
                step=5000,
                key="sim_extra_miles",
            )
        sim_go = st.form_submit_button("Simulate", use_container_width=True)

    if sim_go:
        try:
            sim_mil = int(p["mileage"] + extra_miles)
            # The pricing model encodes age through vehicle year, so simulating
            # a future sale means holding the same car for more years.
            sim_vehicle_year = int(p["year"] - years_to_wait)
            sl, sm, sh = predict.get_price_range(
                make=p["make"], model=p["model"], year=sim_vehicle_year,
                mileage=sim_mil, body_type=p["body"], drive_type=p["drive"],
                zipcode=p.get("zip", ""), engine=p.get("engine", ""),
                trim=p.get("trim", ""),
            )
            st.caption(
                "Scenario: sell {} with {:,} additional miles.".format(
                    "now" if years_to_wait == 0 else "in {} year{}".format(years_to_wait, "" if years_to_wait == 1 else "s"),
                    extra_miles,
                )
            )
            sc1, sc2, sc3 = st.columns(3)
            for col, lbl, orig, new in [
                (sc1, "Competitive", low, sl),
                (sc2, "Fair Market", mid, sm),
                (sc3, "Premium", high, sh),
            ]:
                diff = new - orig
                if diff > 0:
                    delta_text = "+{}".format(fmt(diff))
                elif diff < 0:
                    delta_text = "-{}".format(fmt(abs(diff)))
                else:
                    delta_text = "No change"
                col.metric(
                    lbl,
                    fmt(new),
                    delta=delta_text,
                    delta_color="normal",
                )
        except Exception as e:
            st.error(f"Simulation error: {e}")

    st.divider()

    st.subheader("💡 Pricing Tips")
    tips = [
        ("Start at Fair Market",
         f"List around {fmt(mid)} to attract serious buyers while leaving negotiation room."),
        ("Competitive for Speed",
         f"Price at {fmt(low)} for faster interest and quicker sale."),
    ]
    if car_age <= 3:
        tips.append(("Consider Selling Soon",
                     "Your car is in the steep depreciation phase. Selling sooner preserves more value."))
    else:
        tips.append(("Timing Flexibility",
                     "Your car's depreciation has stabilized. You have more flexibility on timing."))
    try:
        if not region_deals.empty:
            best_r = region_deals.iloc[0]
            tips.append(("Regional Insight",
                         f"Cars tend to sell cheapest in {best_r['region']} "
                         f"({best_r['deal_label']}). If buyers compare, price competitively."))
    except Exception:
        pass

    for title, desc in tips:
        st.markdown(f"**{title}** — {desc}")


def page_buyer_dash():
    p = st.session_state.get("profile")
    if not p:
        nav_to("role"); st.rerun(); return

    if st.button("← Start over"):
        nav_to("role"); st.rerun()

    st.title("🔍 Buyer Recommendations")
    st.caption(
        f"Budget: **{fmt(p['budget'])}**"
        + (f" · Body: **{p['body']}**" if p.get("body") else "")
        + (f" · Drive: **{p['drive']}**" if p.get("drive") else "")
        + (f" · Make: **{p['make']}**" if p.get("make") else "")
    )

    st.info(
        "These recommendations are individual historical vehicle records ranked by how far their listed price sits below the model's fair-value estimate."
    )

    st.subheader("🔧 Refine Your Search")
    df = load_reference_data()
    predict = load_predict_module()
    recommend = load_recommend_module()
    body_types = preferred_body_type_options(
        option_list(df, "BodyType", normalizer=recommend._normalize_body_type)
    )
    drive_types = option_list(df, "DriveType", normalizer=recommend._normalize_drive_type)
    makes = option_list(df, "Make", normalizer=predict._normalize_brand, min_count=10)

    with st.form("buyer_filter"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            budget = st.number_input("Budget ($)", min_value=1000, max_value=150000,
                                     value=int(p["budget"]), step=500)
            f_body = st.selectbox("Body Type", ["Any"] + body_types)
        with fc2:
            f_drive = st.selectbox("Drive Type", ["Any"] + drive_types)
            f_make = st.selectbox("Preferred Make", ["Any"] + makes)
        with fc3:
            max_mil = st.number_input("Max Mileage (0 = no limit)", min_value=0,
                                      max_value=300000, value=0, step=5000)
            min_yr = st.number_input("Min Year (0 = no limit)", min_value=0,
                                     max_value=2026, value=0)
        search = st.form_submit_button("Update Results", use_container_width=True, type="primary")

    b = budget if search else p["budget"]
    fb = (f_body if f_body != "Any" else None) if search else p.get("body")
    fd = (f_drive if f_drive != "Any" else None) if search else p.get("drive")
    fm = (f_make if f_make != "Any" else None) if search else p.get("make")
    mm = (max_mil if max_mil > 0 else None) if search else p.get("max_mil")
    my = (min_yr if min_yr > 0 else None) if search else p.get("min_yr")
    zp = p.get("zip")

    try:
        results = get_cached_recommend(
            budget=float(b),
            body_type=fb, drive_type=fd, make=fm,
            max_mileage=mm, min_year=my,
            zipcode=zp, top_n=10,
        )
    except Exception as e:
        st.error(f"Recommendation error: {e}")
        return

    if not results:
        st.warning("No matches found. Try increasing your budget or loosening filters.")
        return

    st.divider()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Results", len(results))
    avg_p = sum(r["listing_price"] for r in results) // len(results)
    s2.metric("Avg Listing Price", fmt(avg_p))
    best_vp = max(r.get("avg_value_pct", 0) for r in results)
    s3.metric("Best Value", f"{best_vp:.0f}% below")
    high_conf = sum(1 for r in results if r.get("confidence") == "High")
    s4.metric("Strong Data Support", high_conf)

    st.caption("Data support reflects how many comparable sales exist for that make/model in the dataset.")

    st.divider()

    st.subheader("📋 Top 10 Recommendations")
    st.write("These are the listings that currently look most attractive for your filters and budget.")

    for i in range(0, len(results), 2):
        cols = st.columns(2)
        for col, rec in zip(cols, results[i:i + 2]):
            with col:
                rank = i + results[i:i + 2].index(rec) + 1
                conf = rec.get("confidence", "—")
                data_support = rec.get("data_support", rec.get("confidence_help", conf))
                conf_icon = "🟢" if conf == "High" else "🟡" if conf == "Medium" else "🔴"
                vp = rec.get("avg_value_pct", 0)
                deal_icon = "🟢" if vp >= 12 else "🔵" if vp >= 5 else "🟡" if vp >= 0 else "🔴"
                deal = "Excellent Value" if vp >= 12 else "Good Value" if vp >= 5 else "Fair Value" if vp >= 0 else "Above Market"

                try:
                    typical_year = int(str(rec.get("year_range", "")).split("-")[0])
                except Exception:
                    typical_year = None

                with st.container(border=True):
                    img_url = get_cached_image(rec["title"], typical_year)
                    if img_url:
                        st.image(img_url, use_container_width=True)
                    else:
                        st.markdown(
                            "<div style='height:180px;background:rgba(128,128,128,0.1);"
                            "display:flex;align-items:center;justify-content:center;"
                            "color:gray;border-radius:6px;font-size:13px;'>No image</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown(f"**#{rank} — {rec['title']}**")
                    st.caption(f"{deal_icon} {deal} · {conf_icon} {data_support}")
                    mc1, mc2 = st.columns(2)
                    mc1.write(f"📅 **{rec.get('year_range', '—')}**")
                    mc2.write(f"🛣️ **{rec.get('typical_mileage', 0):,} mi**")
                    mc1.write(f"🚗 {rec.get('body_type', '—')}")
                    mc2.write(f"⚙️ {rec.get('drive_type', '—')}")
                    pc1, pc2 = st.columns(2)
                    pc1.metric("Listing Price", fmt(rec["listing_price"]))
                    pc2.metric("Fair Value", fmt(rec["predicted_fair"]))
                    if vp > 0:
                        st.markdown(f"↓ **Estimated savings: {vp:.1f}% below fair value**")
                    if rec.get("reason"):
                        st.caption(f"💬 _{rec['reason']}_")

    st.divider()

    st.subheader("🗺️ Regional Best Deals")
    st.caption("Across the full dataset, these regions tend to offer stronger prices relative to model-estimated fair value.")

    try:
        region_deals = get_cached_region_deals(top_n=6)

        if not region_deals.empty:
            fig_br = go.Figure(go.Bar(
                x=region_deals["region"],
                y=region_deals["price_advantage"],
                marker_color=[
                    "#2fc872" if pa > 0 else "#e85050"
                    for pa in region_deals["price_advantage"]
                ],
                text=region_deals["deal_label"],
                textposition="outside",
            ))
            fig_br.update_layout(
                height=300, margin=dict(t=20, b=40),
                yaxis=dict(tickprefix="$", title="Avg Savings vs Fair Value"),
            )
            st.plotly_chart(fig_br, use_container_width=True)

            st.info(f"💡 **{region_deals.iloc[0]['region']}** has the best deals — "
                    f"{region_deals.iloc[0]['deal_label']}.")
        else:
            st.info("Not enough data for regional analysis.")
    except Exception as e:
        st.warning(f"Regional analysis unavailable: {e}")

    st.divider()

    st.subheader("⚖️ Side-by-Side Comparison")
    st.caption("Shortlist two or three cars to compare price, value gap, mileage, and data support.")

    titles = [r["title"] for r in results]
    selected = st.multiselect("Choose cars to compare", titles, max_selections=3)

    if len(selected) >= 2:
        compare = [r for r in results if r["title"] in selected]
        comp = {"Metric": [
            "Price", "Fair Value", "Value Gap", "Year Range",
            "Mileage", "Body", "Drive", "Confidence", "Samples",
        ]}
        for c in compare:
            vp = c.get("avg_value_pct", 0)
            comp[c["title"]] = [
                fmt(c["typical_price"]), fmt(c["predicted_fair"]),
                f"{vp:.1f}%", c.get("year_range", "—"),
                f"{c.get('typical_mileage', 0):,} mi",
                c.get("body_type", "—"), c.get("drive_type", "—"),
                c.get("confidence", "—"), c.get("sample_count", "—"),
            ]
        st.dataframe(pd.DataFrame(comp).set_index("Metric"), use_container_width=True)
    elif selected:
        st.caption("Select at least 2 cars to compare.")


def main():
    st.set_page_config(page_title="CarPrice", page_icon="🚗", layout="wide")

    st.markdown(
        "<style>"
        "div[data-testid='stHorizontalBlock']"
        "{ align-items: stretch; }"
        "div[data-testid='stHorizontalBlock'] > div[data-testid='stColumn']"
        "{ display: flex; flex-direction: column; }"
        "div[data-testid='stHorizontalBlock'] > div[data-testid='stColumn']"
        " > div[data-testid='stVerticalBlockBorderWrapper']"
        "{ flex: 1; }"
        "</style>",
        unsafe_allow_html=True,
    )

    if "page" not in st.session_state:
        st.session_state["page"] = "role"

    page = st.session_state["page"]

    if page == "role":
        page_role()
    elif page == "profile":
        page_profile()
    elif page == "seller_dash":
        page_seller_dash()
    elif page == "buyer_dash":
        page_buyer_dash()
    else:
        page_role()


if __name__ == "__main__":
    main()
