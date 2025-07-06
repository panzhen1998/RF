#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


data = pd.read_csv('RF.csv')
df = pd.DataFrame(data)


# In[3]:


X = df.drop(['target'], axis=1)
y = df['target']


# In[4]:


target = 'target'
features = df.columns.drop(target)
print(data["target"].value_counts()) # 顺便查看一下样本是否平衡
np.random.seed(seed=5)


# In[5]:


# df = shuffle(df)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=0)


# In[6]:


import xgboost as xgb  # 确保使用正常空格
from sklearn.model_selection import GridSearchCV


# In[7]:


# 修正后的 XGBoost 模型参数（使用正常空格）
params_xgb = {
    'learning_rate': 0.02,            # 学习率，控制每一步的步长，用于防止过拟合。典型值范围：0.01 - 0.1
    'booster': 'gbtree',              # 提升方法，这里使用梯度提升树（Gradient Boosting Tree）
    'objective': 'binary:logistic',   # 损失函数，这里使用逻辑回归，用于二分类任务
    'max_leaves': 127,                # 每棵树的叶子节点数量，控制模型复杂度
    'verbosity': 1,                   # 控制 XGBoost 输出信息的详细程度
    'seed': 42,                       # 随机种子，用于重现模型的结果
    'nthread': -1,                    # 并行运算的线程数量，-1表示使用所有CPU核心
    'colsample_bytree': 0.6,          # 每棵树随机选择的特征比例
    'subsample': 0.7,                 # 每次迭代时随机选择的样本比例
    'eval_metric': 'logloss'          # 评价指标，使用对数损失
}


# In[8]:


model_xgb = xgb.XGBClassifier(**params_xgb)


# In[9]:


# 修正后的参数网格（使用正常空格）
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],  # 树的数量
    'max_depth': [3, 4, 5, 6, 7],               # 树的深度
    'learning_rate': [0.01, 0.02, 0.05, 0.1],   # 学习率
}


# In[10]:


from sklearn.model_selection import GridSearchCV
import joblib

# 使用GridSearchCV进行网格搜索和k折交叉验证（使用正常空格）
grid_search = GridSearchCV(
    estimator=model_xgb,
    param_grid=param_grid,
    scoring='neg_log_loss',  # 评价指标为负对数损失
    cv=5,                    # 5折交叉验证
    n_jobs=-1,               # 并行计算
    verbose=1                # 输出详细进度信息
)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最优参数
print("Best parameters found: ", grid_search.best_params_)
print("Best Log Loss score: ", -grid_search.best_score_)

# 使用最优参数训练模型
best_model = grid_search.best_estimator_

# 保存模型
joblib.dump(best_model, 'XGBoost.pkl')


# In[11]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# 加载模型（确保路径正确）
model_path = os.path.join(os.path.dirname(__file__), 'XGBoost.pkl')
model = joblib.load(model_path)

# 定义选项和特征
N_options = {0: 'No lymph node metastasis (0)', 1: 'Lymph node metastasis (1)'}
T_options = {1: 'cT1 (1)', 2: 'cT2 (2)', 3: 'cT3 (3)', 4: 'cT4 (4)'}
SINIT_options = {1: 'SINIT1 (1)', 2: 'SINIT2 (2)', 3: 'SINIT3 (3)'}
feature_names = ["SINIT", "N", "SIZE", "T"]

# Streamlit 界面
st.title("LARC Disease Predictor")
SIZE = st.number_input("Tumor size", min_value=0.1, max_value=10.0, value=0.1)
N = st.selectbox("cN status:", options=[0, 1], format_func=lambda x: N_options[x])
T = st.selectbox("cT stage:", options=list(T_options.keys()), format_func=lambda x: T_options[x])
SINIT = st.selectbox("SINIT:", options=list(SINIT_options.keys()), format_func=lambda x: SINIT_options[x])

# 预测逻辑
if st.button("Predict"):
    try:
        # 转换为 DataFrame 确保格式匹配
        features = pd.DataFrame([[SINIT, N, T, SIZE]], columns=feature_names)
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]
        
        # 显示结果
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Probability:** {predicted_proba[predicted_class]:.1%}")
        
        # SHAP 可视化（改用 summary_plot 更稳定）
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        
        st.subheader("Feature Importance (SHAP)")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, features, plot_type="bar", show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

