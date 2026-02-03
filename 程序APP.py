import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ======================
# Global plotting style (Nature / Lancet-like)
# ======================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 1,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

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
st.title("Clinical Risk Prediction Tool for AML Patients Undergoing allo-HSCT")
st.markdown(
    """
    **Purpose:**  
    Estimate the individual risk of 2-year mortality after allogeneic hematopoietic stem cell transplantation  
    and provide transparent, patient-level model explanation.
    """
)

st.subheader("Patient Characteristics")

feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            f"{feature}",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    else:
        value = st.selectbox(
            f"{feature}",
            options=properties["options"],
            index=properties["options"].index(properties["default"]),
        )
    feature_values.append(value)

X_input = pd.DataFrame([feature_values], columns=feature_names)

# ======================
# Prediction
# ======================
if st.button("Calculate Risk"):

    proba = model.predict_proba(X_input)[0][1] * 100

    # ---- Risk stratification ----
    if proba < 20:
        risk_group = "Low risk"
        color = "green"
    elif proba < 50:
        risk_group = "Intermediate risk"
        color = "orange"
    else:
        risk_group = "High risk"
        color = "red"

    # ---- Display risk score ----
    st.markdown("### Predicted 2-Year Mortality Risk")

    st.markdown(
        f"""
        <div style="border-left:6px solid {color}; padding:12px;">
        <span style="font-size:28px; font-weight:bold;">{proba:.1f}%</span><br>
        <span style="font-size:18px; color:{color};"><b>{risk_group}</b></span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ======================
    # SHAP explanation
    # ======================
    st.markdown("### Individual Risk Explanation (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)

    if isinstance(shap_values, list):
        shap_values_class = shap_values[1]
    else:
        shap_values_class = shap_values

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP value": shap_values_class[0]
    }).sort_values(by="SHAP value", key=np.abs, ascending=True)

    # ======================
    # Nature-style SHAP bar plot
    # ======================
    fig, ax = plt.subplots(figsize=(5, 5))

    colors = shap_df["SHAP value"].apply(
        lambda x: "#d62728" if x > 0 else "#1f77b4"
    )

    ax.barh(
        shap_df["Feature"],
        shap_df["SHAP value"],
        color=colors
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on mortality risk)")
    ax.set_title("Key Contributors to Individual Risk")

    plt.tight_layout()
    plt.savefig("shap_bar_nature.png", dpi=600)
    plt.close()

    st.image("shap_bar_nature.png")



