import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="SaaS Profit Predictor",
    page_icon="💰",
    layout="wide"
)

# Application Title and Intro
st.title("📊 SaaS Sales Profit Prediction Dashboard")
st.markdown("""
This interactive web dashboard leverages your trained **DecisionTreeRegressor with RFE** machine learning pipeline 
to project transaction profits based on localized geographical, client, product, and financial attributes.
""")
st.markdown("---")

# Cached model loading function
@st.cache_resource
def load_ml_pipeline():
    try:
        # Attempts to load your exported pipeline object
        return joblib.load("saas_pipeline.pkl")
    except FileNotFoundError:
        st.error("⚠️ **Model File Error:** `saas_pipeline.pkl` was not found in the root directory. Please place the file in the same folder as this script.")
        return None

pipeline = load_ml_pipeline()

if pipeline:
    # Sidebar context panel
    st.sidebar.header("⚙️ Model Blueprint Meta")
    st.sidebar.success("✅ Model loaded successfully!")
    st.sidebar.info("""
    **Pipeline Flow:**
    1. Categorical Encoding (`OneHotEncoder`)
    2. Numerical Scaling (`StandardScaler`)
    3. Feature Selection (`RFE` w/ LinearRegression)
    4. Estimation (`DecisionTreeRegressor`)
    """)

    # Grid breakdown layout for cleaner inputs
    st.subheader("🛠️ Transaction Feature Variables")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌍 Geography & Client")
        country = st.selectbox("Country", ["Ireland", "United States", "Germany", "United Kingdom", "France", "Canada"])
        city = st.selectbox("City", ["Dublin", "New York City", "Stuttgart", "Milwaukee", "Hamburg", "Dallas", "London", "Paris"])
        region = st.selectbox("Region", ["EMEA", "AMER", "APAC"])
        subregion = st.selectbox("Subregion", ["UKIR", "NAMER", "EU-WEST", "APAC-S"])
        customer = st.selectbox("Customer", ["Chevron", "Phillips 66", "Royal Dutch Shell", "Johnson & Johnson", "American Express", "Comcast", "Walmart", "Apple"])

    with col2:
        st.markdown("### 💼 Segmentation & Product")
        industry = st.selectbox("Industry", ["Energy", "Healthcare", "Finance", "Communications", "Technology", "Retail"])
        segment = st.selectbox("Segment", ["SMB", "Strategic", "Enterprise", "B2B", "B2C"])
        product = st.selectbox("Product", [
            "Marketing Suite", "FinanceHub", "ContactMatcher", 
            "Marketing Suite - Gold", "SaaS Connector Pack", 
            "Site Analytics", "Support", "OneView"
        ])
        
    with col3:
        st.markdown("### 📈 Financials & Date Strategy")
        sales = st.number_input("Sales Revenue ($)", min_value=0.0, value=261.96, step=10.0, format="%.4f")
        quantity = st.number_input("Quantity Sold", min_value=1, value=2, step=1)
        discount = st.slider("Discount Applied", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        
        # Chronological splits 
        year = st.number_input("Year", min_value=2020, max_value=2035, value=2022, step=1)
        month = st.slider("Month", min_value=1, max_value=12, value=11)
        day = st.slider("Day", min_value=1, max_value=31, value=9)

    # Structuring user choices into matching DataFrame structure 
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
    
    # Live preview framework
    st.write("### 🔍 Input Blueprint (Dataframe View)")
    st.dataframe(input_df)

    # Executing calculation logic
    if st.button("🚀 Run Profit Prediction", type="primary"):
        with st.spinner("Processing framework steps..."):
            try:
                # Target the pipeline prediction node directly
                prediction = pipeline.predict(input_df)[0]
                
                # Visual output presentation
                st.markdown("### 🎯 Model Prediction Results")
                metric_col, notes_col = st.columns([1, 2])
                
                if prediction >= 0:
                    metric_col.metric(label="Projected Gross Profit", value=f"${prediction:,.4f}")
                    notes_col.success("✨ Safe margin! The current settings predict a net positive profit outcome.")
                    st.balloons()
                else:
                    metric_col.metric(label="Projected Gross Loss", value=f"${prediction:,.4f}", delta_color="inverse")
                    notes_col.warning("⚠️ Warning! The current pricing variables generate a financial system net-loss.")
                    
            except Exception as e:
                st.error(f"Execution handling error: {e}")
                st.info("Ensure the category strings fall perfectly within your `saas_pipeline.pkl` bounds or that your encoder setup tolerates unknown strings.")