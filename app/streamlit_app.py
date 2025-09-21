import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from preprocess import RetailPreprocessor
from huggingface_hub import hf_hub_download
import requests
import os


# ------------------------
# Streamlit Intialization
# ------------------------
st.set_page_config(
    page_title="Retail Demand Predictor", 
    page_icon="📈",
    layout="wide"
)

# ------------------------
# Load pipeline
# ------------------------
@st.cache_resource
def load_pipeline():
    model_path = hf_hub_download(
        repo_id="oladosularinde/retail-demand",
        filename="demand_forecast_pipeline.pkl"
    )

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    return pipeline

pipeline = load_pipeline()

# ------------------------
# Streamlit UI
# ------------------------
st.title("📈 Retail Demand Forecasting App")
st.write("Predict retail demand using historical sales, promotions, and external factors.")

# Tabs
tab1, tab2 = st.tabs(["📂 Batch Predictions", "🛠 Single Prediction"])

# ------------------------
# Tab 1: Batch Predictions
# ------------------------
with tab1:
    st.subheader("Upload Your Data")
    required_columns = [
        "Date", "Store ID", "Product ID", "Category", "Region",
        "Inventory Level", "Units Sold", "Units Ordered",
        "Price", "Discount", "Weather Condition", "Promotion",
        "Competitor Pricing", "Seasonality", "Epidemic"
    ]

    uploaded_file = st.file_uploader(
        "Upload CSV with raw retail data.",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview", df.head())

        # Check for missing columns
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            st.error(
                f"⚠️ The uploaded file is missing these required columns:\n\n- " +
                "\n- ".join(missing_cols)
            )
            st.stop()
        else:
            if st.button("Run Predictions"):
                # Predict
                preds_log = pipeline.predict(df)
                preds_real = np.expm1(preds_log)

                df["Predicted Demand"] = np.ceil(preds_real).astype(int)
                st.success("Predictions complete ✅")
                st.write(df.head())

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="demand_predictions.csv",
                    mime="text/csv"
                )

# ------------------------
# Tab 2: Single Prediction
# ------------------------
with tab2:
    st.subheader("Enter Details for One Prediction")

    # Numeric inputs
    price = st.number_input("Price", min_value=0.0, value=50.0)
    discount = st.number_input("Discount (%)", min_value=0.0, value=5.0)
    units_sold = st.number_input("Units Sold", min_value=0, value=10)
    units_ordered = st.number_input("Units Ordered", min_value=0, value=15)
    competitor_price = st.number_input("Competitor Pricing", min_value=0.0, value=48.0)
    inventory_level = st.number_input("Inventory Level", min_value=0, value=100)

    # Categorical inputs
    category = st.selectbox("Category", ["Groceries", "Clothing", "Electronics", "Furniture", "Toys"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy", "Snowy"])
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])
    promotion_input = st.selectbox("Running promotions?", ["No", "Yes"])
    epidemic_input = st.selectbox("Is there an epidemic?", ["No", "Yes"])
    product_id = st.text_input("Product ID", "P0001")

    if st.button("Predict Demand"):
        # Build one-row dataframe
        promotion = 1 if promotion_input == "Yes" else 0
        epidemic = 1 if epidemic_input == "Yes" else 0
        single_data = pd.DataFrame([{
            "Price": price,
            "Discount": discount,
            "Units Sold": units_sold,
            "Units Ordered": units_ordered,
            "Competitor Pricing": competitor_price,
            "Inventory Level": inventory_level,
            "Category": category,
            "Region": region,
            "Weather Condition": weather,
            "Seasonality": season,
            "Promotion": promotion,
            "Epidemic": epidemic,
            "Date": str(datetime.now().date()),
            "Store ID": "S0001",
            "Product ID": product_id
        }])

        pred_log = pipeline.predict(single_data)
        pred_real = np.expm1(pred_log)[0]
        pred = np.ceil(pred_real).astype(int)
        st.metric("Predicted Demand", f"{pred} units")
