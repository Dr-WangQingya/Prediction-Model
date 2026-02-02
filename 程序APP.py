import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('lgbm.pkl')

# 特征定义：字典键是显示名称，内部包含模型中的实际名称和参数
feature_definitions = {
    "MRD": {
        "model_name": "MRD",
        "type": "categorical", 
        "options": [0, 1], 
        "default": 1
    },
    "Disease stage before transplantation": {
        "model_name": "disease_stage",
        "type": "categorical", 
        "options": [1, 2, 3], 
        "default": 1
    },
    "PLT graft failure": {
        "model_name": "PLT_graft_failure",
        "type": "categorical", 
        "options": [0, 1], 
        "default": 1
    },
    "MNC": {
        "model_name": "MNC",
        "type": "numerical", 
        "min": 4.521, 
        "max": 35.021, 
        "default": 11.161
    },
    "PLT graft time": {
        "model_name": "PLT_time",  # 模型中的实际名称
        "type": "numerical", 
        "min": 8.00, 
        "max": 81.00, 
        "default": 14
    },
    "CD34": {
        "model_name": "CD34",
        "type": "numerical", 
        "min": 1.311, 
        "max": 22.411, 
        "default": 6.391
    },
    "HCT-CI": {
        "model_name": "HCT_CI",
        "type": "categorical", 
        "options": [0, 1, 2, 3, 4, 5], 
        "default": 0
    },
    "WBC count at diagnose": {
        "model_name": "WBC",
        "type": "numerical", 
        "min": 0.710, 
        "max": 369.080, 
        "default": 0.98
    },
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
user_inputs = {}
model_feature_order = []  # 保存模型特征顺序

for display_name, properties in feature_definitions.items():
    model_name = properties["model_name"]
    model_feature_order.append(model_name)  # 保存模型特征顺序
    
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{display_name} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
            key=display_name  # 使用显示名作为Streamlit的key
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{display_name} (Select a value)",
            options=properties["options"],
            key=display_name  # 使用显示名作为Streamlit的key
        )
    
    user_inputs[model_name] = value

# 转换为模型输入格式（按照模型特征顺序）
feature_values = [user_inputs[model_name] for model_name in model_feature_order]
features = np.array([feature_values])

# 创建用于显示的DataFrame（使用显示名）
display_df = pd.DataFrame(
    [list(user_inputs.values())], 
    columns=[display_name for display_name in feature_definitions.keys()]
)

# 创建用于模型的DataFrame（使用模型特征名）
model_df = pd.DataFrame(
    [list(user_inputs.values())], 
    columns=model_feature_order
)

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(model_df)[0]
    predicted_proba = model.predict_proba(model_df)[0]
    
    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100
    
    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on the above values, the probability of death within 2 years following allogeneic hematopoietic stem cell transplantation is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")
    
    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(model_df)
    
    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    
    # 创建带有显示名的SHAP图
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index] if len(shap_values.shape) > 2 else shap_values[class_index],
        display_df,  # 使用显示名的DataFrame
        matplotlib=True,
    )
    
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
    
    # 可选：显示输入值的摘要
    st.subheader("Input Summary")
    for display_name, properties in feature_definitions.items():
        model_name = properties["model_name"]
        st.write(f"{display_name}: {user_inputs[model_name]}")
