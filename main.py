import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Custom dataset loaded ✅")
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")
        df = load_demo_data()
else:
    df = load_demo_data()
    st.sidebar.info("Using demo dataset")

# ----------------- Define Features -----------------
feature_cols = [c for c in ["Age","JobLevel","MonthlyIncome","YearsAtCompany"] if c in df.columns]
if not feature_cols:
    st.error("❌ No valid features found in dataset. Please include Age, JobLevel, MonthlyIncome, or YearsAtCompany.")
    st.stop()

# ----------------- Train Model -----------------
X = df[feature_cols]
y = df["Attrition"] if "Attrition" in df.columns else [0]*len(df)  # dummy if missing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# ----------------- Sidebar Menu -----------------
menu = st.sidebar.radio("Choose a section", ["Overview","EDA","Attrition Dashboard","Attrition Prediction"])

# ----------------- Overview -----------------
if menu == "Overview":
    st.subheader("📌 Project Overview")
    st.write("Upload your HR data or use the demo. The model predicts attrition risk.")
    st.metric("Model Accuracy (Test Set)", f"{acc:.2%}")

# ----------------- EDA -----------------
elif menu == "EDA":
    st.subheader("🔍 Exploratory Data Analysis")
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    if len(numeric_cols) > 1:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=True, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for correlation heatmap.")

    st.write("### Summary Statistics")
    st.write(df.describe())

# ----------------- Dashboard -----------------
elif menu == "Attrition Dashboard":
    if "Attrition" not in df.columns:
        st.warning("⚠️ Attrition column not found in dataset.")
    else:
        st.subheader("📊 Attrition Risk Dashboard")
        if "Department" in df.columns:
            dept_chart = df.groupby("Department")["Attrition"].mean().reset_index()
            st.bar_chart(dept_chart.set_index("Department"))
        else:
            st.info("No Department column found, skipping department analysis.")

# ----------------- Prediction -----------------
elif menu == "Attrition Prediction":
    st.subheader("🤖 Predict Employee Attrition")
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
