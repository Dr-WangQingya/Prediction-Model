# --------------------------------------------------
# 2. SHAP explanation (ROBUST for LightGBM binary)
# --------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_input)
expected_value = explainer.expected_value

# -------- robust handling for binary classifier -----
if isinstance(expected_value, (list, np.ndarray)):
    # rare case: explicit class outputs
    base_logit = expected_value[1]
    shap_logit = shap_values[1][0]
else:
    # most common LightGBM binary case
    base_logit = expected_value
    shap_logit = shap_values[0]

# -------- consistency check (logit → probability) ----
fx_logit = base_logit + shap_logit.sum()
fx_prob = 1 / (1 + np.exp(-fx_logit))

# -------- Force plot (logit space, mathematically correct) ----
shap.force_plot(
    base_logit,
    shap_logit,
    X_input.iloc[0, :],
    matplotlib=True,
    show=False,
    figsize=(15, 4)
)

plt.tight_layout()
plt.savefig("shap_force_plot.png", dpi=600, bbox_inches="tight")
plt.close()

st.image("shap_force_plot.png")
