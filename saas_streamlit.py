import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="SaaS Profit Prediction & Optimization Platform",
    page_icon="💼",
    layout="wide"
)

# Professional Enterprise Header
st.title("AI-Powered SaaS Profit Prediction & Optimization Platform")
st.caption("Predictive Analytics & Marginal Return Forecasting Engine for Enterprise Sales Execution")
st.markdown("---")

# Cached model loading function
@st.cache_resource
def load_ml_pipeline():
    try:
        return joblib.load("saas_pipeline.pkl")
    except FileNotFoundError:
        st.error("**Model File Error:** The predictive pipeline file `saas_pipeline.pkl` was not found in the root directory. Please place the serialized pipeline in the same folder as this application script.")
        return None

pipeline = load_ml_pipeline()

if pipeline:
    # Input parameter matrix layout
    st.subheader("Transaction Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Geography & Client")
        country = st.selectbox("Country", ["Ireland", "United States", "Germany", "United Kingdom", "France", "Canada"])
        city = st.selectbox("City", ["Dublin", "New York City", "Stuttgart", "Milwaukee", "Hamburg", "Dallas", "London", "Paris"])
        region = st.selectbox("Region", ["EMEA", "AMER", "APAC"])
        subregion = st.selectbox("Subregion", ["UKIR", "NAMER", "EU-WEST", "APAC-S"])
        customer = st.selectbox("Customer", ["Chevron", "Phillips 66", "Royal Dutch Shell", "Johnson & Johnson", "American Express", "Comcast", "Walmart", "Apple"])

    with col2:
        st.markdown("### Segmentation & Product")
        industry = st.selectbox("Industry", ["Energy", "Healthcare", "Finance", "Communications", "Technology", "Retail"])
        segment = st.selectbox("Segment", ["SMB", "Strategic", "Enterprise", "B2B", "B2C"])
        product = st.selectbox("Product", [
            "Marketing Suite", "FinanceHub", "ContactMatcher", 
            "Marketing Suite - Gold", "SaaS Connector Pack", 
            "Site Analytics", "Support", "OneView"
        ])
        
    with col3:
        st.markdown("### Financials & Temporal Logic")
        sales = st.number_input("Gross Revenue ($)", min_value=0.0, value=261.96, step=10.0, format="%.4f")
        quantity = st.number_input("Units / Volume", min_value=1, value=2, step=1)
        discount = st.slider("Contractual Discount Applied", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        
        # Chronological splits 
        year = st.number_input("Fiscal Year", min_value=2020, max_value=2035, value=2022, step=1)
        month = st.slider("Fiscal Month", min_value=1, max_value=12, value=11)
        day = st.slider("Fiscal Day", min_value=1, max_value=31, value=9)

    # Structuring feature arrays matching the upstream pipeline schema
    input_df = pd.DataFrame({
        "Country": [country],
        "City": [city],
        "Region": [region],
        "Subregion": [subregion],
        "Customer": [customer],
        "Industry": [industry],
        "Segment": [segment],
        "Product": [product],
        "Sales": [sales],
        "Quantity": [float(quantity)],
        "Discount": [discount],
        "Year": [int(year)],
        "Month": [int(month)],
        "Day": [int(day)]
    })

    st.markdown("---")
    
    # Input Blueprint Dataframe Framework
    st.write("### Input Vector Profile (DataFrame Preview)")
    st.dataframe(input_df)

    # Model Evaluation Pipeline Execution
    if st.button("Execute Predictive Optimization Model", type="primary"):
        with st.spinner("Processing architectural pipeline transformations..."):
            try:
                # Direct prediction calculation
                prediction = pipeline.predict(input_df)[0]
                
                st.markdown("---")
                st.markdown("### Model Optimization Output")
                metric_col, notes_col = st.columns([1, 2])
                
                if prediction >= 0:
                    metric_col.metric(label="Projected Net Margin (Profit)", value=f"${prediction:,.4f}")
                    notes_col.success("The structured deal matrices generate a favorable, profitable margin. Risk validation cleared.")
                else:
                    metric_col.metric(label="Projected Net Margin (Loss)", value=f"${prediction:,.4f}", delta_color="inverse")
                    notes_col.warning("The evaluated configuration results in a financial system deficit. Adjust pricing structures, discount parameters, or volume scaling inputs.")
                    
            except Exception as e:
                st.error(f"Runtime Pipeline Exception Encountered: {e}")