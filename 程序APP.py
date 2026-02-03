import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ======================
# Global plotting style
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
st.title(
    "Prognostic Model for Acute Myeloid Leukemia Patients "
    "Undergoing Allogeneic Hematopoietic Stem Cell Transplantation"
)

st.subheader("Enter the following feature values:")

feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            feature,
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    else:
        value = st.selectbox(
            feature,
            options=properties["options"],
            index=properties["options"].index(properties["default"]),
        )
    feature_values.append(value)

X_input = pd.DataFrame([feature_values], columns=feature_names)

# ======================
# Prediction & SHAP
# ======================
if st.button("Predict"):

    # --------------------------------------------------
    # 1. Predicted probability (FIXED: class = 1 only)
    # --------------------------------------------------
    predicted_proba = model.predict_proba(X_input)[0]
    probability = predicted_proba[1] * 100   # class 1 = death within 2 years

    # --------------------------------------------------
    # 2. Display prediction text (original wording)
    # --------------------------------------------------
    text = (
        f"Based on the above values, the probability of death within 2 years "
        f"following allogeneic hematopoietic stem cell transplantation is "
        f"{probability:.2f}%"
    )

    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.text(
        0.5, 0.5,
        text,
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

    # --------------------------------------------------
    # 3. SHAP explanation (Force plot, SAME output node)
    # --------------------------------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)

    #expected_value = explainer.expected_value[1]
    #shap_values_class = shap_values[1][0]

    base_value_prob = 1 / (1 + np.exp(-explainer.expected_value))  # 基准概率
    shap_values_prob = shap_values[sample_index] / (1 + np.exp(-explainer.expected_value))  # 近似转换
    shap.force_plot(base_value_prob, shap_values_prob, 
                X_test.iloc[sample_index], 
                matplotlib=True, show=False,figsize=(15, 4)) # 这里设置图形大小))

    
    plt.tight_layout()
    plt.savefig("shap_force_plot.png", dpi=600, bbox_inches="tight")
    plt.close()

    st.image("shap_force_plot.png")


