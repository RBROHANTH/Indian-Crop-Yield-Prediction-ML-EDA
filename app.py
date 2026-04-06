import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# ── Load Model & Encoder ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "model_rf.pkl"))
    le    = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    return model, le

model, le = load_model()

# ── Header ───────────────────────────────────────────────────
st.title("🌾 Crop Yield Prediction — India")
st.markdown("**22ADX01 Data Visualization | Micro Project | Kongu Engineering College**")
st.markdown("---")

# ── Layout: two columns ──────────────────────────────────────
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📋 Enter Input Features")

    crop = st.selectbox(
        "Crop Type",
        options=list(le.classes_),
        help="Select the crop you want to predict yield for"
    )

    year = st.slider(
        "Year",
        min_value=1990,
        max_value=2030,
        value=2010,
        step=1
    )

    rainfall = st.number_input(
        "Average Rainfall (mm/year)",
        min_value=0.0,
        max_value=5000.0,
        value=1083.0,
        step=10.0
    )

    pesticides = st.number_input(
        "Pesticides Used (tonnes)",
        min_value=0.0,
        max_value=200000.0,
        value=50000.0,
        step=1000.0
    )

    temperature = st.number_input(
        "Average Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=26.0,
        step=0.1
    )

    predict_btn = st.button("🔍 Predict Yield", use_container_width=True, type="primary")

with col_output:
    st.subheader("📊 Prediction Result")

    if predict_btn:
        crop_encoded = le.transform([crop])[0]

        input_df = pd.DataFrame({
            'Year':                          [year],
            'average_rain_fall_mm_per_year': [rainfall],
            'pesticides_tonnes':             [pesticides],
            'avg_temp':                      [temperature],
            'crop_encoded':                  [crop_encoded]
        })

        predicted_hg = model.predict(input_df)[0]
        predicted_kg = predicted_hg / 100

        st.success("✅ Prediction Complete!")

        m1, m2 = st.columns(2)
        m1.metric("Predicted Yield (hg/ha)", f"{predicted_hg:,.0f}")
        m2.metric("Predicted Yield (kg/ha)", f"{predicted_kg:,.0f}")

        st.markdown("---")
        st.markdown(f"""
        **Summary:**
        - **Crop:** {crop}
        - **Year:** {year}
        - **Rainfall:** {rainfall} mm/year
        - **Pesticides:** {pesticides:,.0f} tonnes
        - **Avg Temperature:** {temperature}°C
        - **Predicted Yield:** `{predicted_hg:,.0f} hg/ha` = `{predicted_kg:,.0f} kg/ha`
        """)

        # ── Feature Importance Chart ──────────────────────────
        st.markdown("---")
        st.subheader("🔍 Feature Importance (Random Forest)")

        feature_names = ['Year', 'Rainfall', 'Pesticides', 'Avg Temp', 'Crop Type']
        importances   = model.feature_importances_
        sorted_idx    = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ['#1F4E79' if i == sorted_idx[0] else '#AED6F1' for i in range(len(feature_names))]
        ax.bar(
            [feature_names[i] for i in sorted_idx],
            [importances[i] for i in sorted_idx],
            color=[colors[i] for i in sorted_idx]
        )
        ax.set_ylabel("Importance Score")
        ax.set_title("What drives yield prediction?")
        ax.tick_params(axis='x', rotation=15)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.info("👈 Fill in the inputs on the left and click **Predict Yield** to see the result.")

        # Show model info when idle
        st.markdown("---")
        st.subheader("ℹ️ Model Information")
        st.markdown("""
        | Property | Value |
        |---|---|
        | **Algorithm** | Random Forest Regressor |
        | **Training Data** | India Crop Yield 1990–2013 |
        | **Records** | 4,048 |
        | **Features** | 5 (Year, Rainfall, Pesticides, Temp, Crop) |
        | **Cross-Val R²** | 0.9800 |
        | **Crops Supported** | 8 |
        """)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey; font-size:13px;'>"
    "Crop Yield Prediction | 22ADX01 Data Visualization | "
    "Kongu Engineering College, Dept. of IT | Rohanth R B</p>",
    unsafe_allow_html=True
)
