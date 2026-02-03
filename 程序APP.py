import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ======================
# Load model
# ======================
model = joblib.load("lgbm.pkl")

# ======================
# Feature configuration
# ======================
feature_ranges = {
    "MRD": {"type": "categorical", "options": [0, 1], "default": 1},
    "Disease stage before transplantation": {"type": "categorical", "options": [1, 2, 3], "default": 1},
    "PLT graft failure": {"type": "categorical", "options": [0, 1], "default": 1},
    "MNC": {"type": "numerical", "min": 4.521, "max": 35.021, "default": 11.161},
    "PLT graft time": {"type": "numerical", "min": 8, "max": 81, "default": 14},
    "CD34": {"type": "numerical", "min": 1.311, "max": 22.411, "default": 6.391},
    "HCT-CI": {"type": "categorical", "options": [0, 1, 2, 3, 4, 5], "default": 0},
    "WBC count at diagnose": {"type": "numerical", "min": 0.710, "max": 369.080, "default": 0.98},
}

feature_names = list(feature_ranges.keys())

# ======================
# Streamlit UI
# ======================
st.title(
    "Prognostic Model for Acute Myeloid Leukemia Patients Undergoing "
    "Allogeneic Hematopoietic Stem Cell Transplantation"
)

st.header("Enter the following feature values:")

feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} – {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    else:
        value = st.selectbox(
            label=f"{feature}",
            options=properties["options"],
            index=properties["options"].index(properties["default"]),
        )
    feature_values.append(value)

# ======================
# Convert to DataFrame
# ======================
X_input = pd.DataFrame([feature_values], columns=feature_names)

# ======================
# Prediction & SHAP
# ======================
if st.button("Predict"):

    # ---- Prediction ----
    predicted_class = int(model.predict(X_input)[1])
    predicted_proba = model.predict_proba(X_input)[1]

    # 明确：1 = 2年内死亡风险
    death_probability = predicted_proba[1] * 100

    # ---- Display prediction text ----
    text = (
        "Based on the above values, the predicted probability of death "
        f"within 2 years following allogeneic hematopoietic stem cell "
        f"transplantation is {death_probability:.2f}%."
    )

    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha="center",
        va="center",
        fontname="Times New Roman",
        transform=ax.transAxes
    )
    ax.axis("off")
    plt.savefig("prediction_text.png", bbox_inches="tight", dpi=300)
    plt.close()
    st.image("prediction_text.png")

    # ======================
    # SHAP explanation
    # ======================
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)

    # LightGBM 二分类：shap_values 为 list
    if isinstance(shap_values, list):
        shap_values_class = shap_values[predicted_class]
        expected_value = explainer.expected_value[predicted_class]
    else:
        shap_values_class = shap_values
        expected_value = explainer.expected_value

    # ======================
    # SHAP force plot (matplotlib)
    # ======================
    plt.figure(figsize=(12, 2))

    shap.force_plot(
        expected_value,
        shap_values_class[0, :],
        X_input.iloc[0, :],
        matplotlib=True,
        show=False
    )

    plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=1200)
    plt.close()

    st.image("shap_force_plot.png")

