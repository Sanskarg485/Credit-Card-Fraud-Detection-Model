import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection App")

@st.cache_data
def load_data():
    df1 = pd.read_csv("C:\\Users\\Sanskar Gupta\\OneDrive\\Desktop\\ML Projects\\Credit Card Fraud Detection Model\\fraudTrain.csv")
    df2 = pd.read_csv("C:\\Users\\Sanskar Gupta\\OneDrive\\Desktop\\ML Projects\\Credit Card Fraud Detection Model\\fraudTest.csv")
    df = pd.concat([df1, df2])
    df.drop(columns=['Unnamed: 0', 'trans_num', 'dob', 'trans_date_trans_time', 'cc_num',
                     'first', 'last', 'street'], inplace=True)
    return df

df = load_data()

# Label Encoding
categorical_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare Features and Target
df.dropna(subset=['is_fraud'], inplace=True)
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model Selection
model_name = st.selectbox("Choose ML Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(max_depth=10)
else:
    model = RandomForestClassifier(n_estimators=100, max_depth=10)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

st.subheader("Model Evaluation Metrics")
col1, col2 = st.columns(2)

with col1:
    st.write("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues")
    st.pyplot(fig)

with col2:
    st.write("Classification Report")
    st.text(classification_report(y_test, y_pred))

st.metric("ROC AUC Score", f"{roc_auc_score(y_test, y_proba):.4f}")

# Manual Prediction
st.subheader("🔍 Predict Fraud for a New Transaction")
sample_input = {}
for col in X.columns:
    dtype = X[col].dtype
    if dtype == 'int64' or dtype == 'float64':
        val = st.number_input(f"{col}", value=float(X[col].mean()))
    else:
        val = st.selectbox(f"{col}", options=list(df[col].unique()))
    sample_input[col] = val

if st.button("Predict"):
    input_df = pd.DataFrame([sample_input])
    # Apply label encoding
    for col in categorical_cols:
        if col in input_df:
            input_df[col] = label_encoders[col].transform([input_df[col]])[0]
    prediction = model.predict(input_df)[0]
    st.success("✅ This transaction is Legitimate." if prediction == 0 else "⚠️ This transaction is Fraudulent!")
