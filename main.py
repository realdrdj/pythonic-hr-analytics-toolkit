import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="HR Analytics Toolkit", layout="wide")
st.title("🧑‍💼 Pythonic HR Analytics Toolkit")

# ----------------- Load Data -----------------
@st.cache_data
def load_demo_data():
    return pd.DataFrame({
        "Age": [25, 30, 45, 40, 28, 35, 50, 38],
        "JobLevel": [1, 2, 3, 2, 1, 4, 3, 2],
        "MonthlyIncome": [3000, 5000, 12000, 8000, 4000, 15000, 9000, 7000],
        "YearsAtCompany": [1, 5, 10, 8, 2, 12, 6, 4],
        "Department": ["Sales","R&D","Sales","HR","Sales","R&D","R&D","HR"],
        "Attrition": [1, 0, 0, 1, 0, 0, 1, 0]
    })

uploaded_file = st.sidebar.file_uploader("📂 Upload your HR dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom dataset loaded successfully ✅")
else:
    df = load_demo_data()
    st.sidebar.info("Using demo dataset")

# ----------------- Preprocess -----------------
def preprocess_for_prediction(df):
    features = [col for col in ["Age","JobLevel","MonthlyIncome","YearsAtCompany"] if col in df.columns]
    X = df[features]
    y = df["Attrition"]
    return train_test_split(X, y, test_size=0.2, random_state=42), features

# ----------------- Train & Evaluate -----------------
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, report

# Train model
(train_X, test_X, train_y, test_y), feature_cols = preprocess_for_prediction(df)
model = train_model(train_X, train_y)
acc, report = evaluate_model(model, test_X, test_y)

# ----------------- Sidebar Menu -----------------
menu = st.sidebar.radio(
    "Choose a section",
    ["Overview", "EDA", "Attrition Dashboard", "Attrition Prediction"]
)

# ----------------- Overview -----------------
if menu == "Overview":
    st.subheader("📌 Project Overview")
    st.markdown("""
    This toolkit demonstrates **practical HR analytics**:
    - Upload your HR dataset (or use demo).
    - Explore workforce trends.
    - Predict employee attrition.
    - Export actionable reports.
    """)
    st.metric("Model Accuracy (Test Set)", f"{acc:.2%}")

# ----------------- EDA -----------------
elif menu == "EDA":
    st.subheader("🔍 Exploratory Data Analysis")
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    if len(numeric_cols) > 1:
        st.write("### Correlation Heatmap (Interactive)")
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)

    st.write("### Summary Statistics")
    st.write(df.describe())

# ----------------- Dashboard -----------------
elif menu == "Attrition Dashboard":
    if "Attrition" not in df.columns:
        st.error("Attrition column missing in dataset.")
    else:
        st.subheader("📊 Attrition Risk Dashboard")

        # Attrition by Department
        if "Department" in df.columns:
            dept_chart = df.groupby("Department")["Attrition"].mean().reset_index()
            fig = px.bar(dept_chart, x="Department", y="Attrition", title="Attrition Rate by Department",
                         labels={"Attrition":"Attrition Rate"})
            st.plotly_chart(fig, use_container_width=True)

        # Attrition by Age Group
        df["AgeGroup"] = pd.cut(df["Age"], bins=[18,30,40,50,60], labels=["18-30","31-40","41-50","51-60"])
        age_chart = df.groupby("AgeGroup")["Attrition"].mean().reset_index()
        fig = px.bar(age_chart, x="AgeGroup", y="Attrition", title="Attrition Rate by Age Group",
                     labels={"Attrition":"Attrition Rate"})
        st.plotly_chart(fig, use_container_width=True)

# ----------------- Prediction -----------------
elif menu == "Attrition Prediction":
    st.subheader("🤖 Predict Employee Attrition")

    # Manual input
    st.markdown("### Test a Hypothetical Employee")
    age = st.number_input("Age", 18, 60, 30)
    job_level = st.slider("Job Level", 1, 5, 2)
    monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
    years_at_company = st.slider("Years at Company", 0, 40, 5)

    input_data = pd.DataFrame([[age, job_level, monthly_income, years_at_company]],
                              columns=feature_cols)

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.write(f"### Prediction: {'⚠️ Attrition Risk' if pred==1 else '✅ No Attrition'}")
    st.write(f"Probability of leaving: **{prob:.2%}**")

    # Bulk predictions
    if uploaded_file:
        st.markdown("### Bulk Prediction on Uploaded Dataset")
        df_pred = df.copy()
        df_pred["Attrition_Predicted"] = model.predict(df_pred[feature_cols])
        df_pred["Attrition_Prob"] = model.predict_proba(df_pred[feature_cols])[:,1]
        st.dataframe(df_pred.head())

        # Download option
        csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", csv, "attrition_predictions.csv", "text/csv")
