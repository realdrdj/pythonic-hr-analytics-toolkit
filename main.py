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

# ----------------- Ensure Attrition Numeric -----------------
if "Attrition" in df.columns and df["Attrition"].dtype == "object":
    df["Attrition"] = df["Attrition"].map({"Yes":1,"No":0,"Y":1,"N":0}).fillna(0).astype(int)

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
    st.markdown("""
    Welcome to the **Pythonic HR Analytics Toolkit**.  
    This tool helps HR professionals, managers, and researchers **analyze employee data and predict attrition risk**.

    ### 🔍 How It Works
    - Upload your **HR dataset (CSV)**, or use the built-in demo dataset.  
    - The system automatically trains a machine learning model to detect attrition patterns.  
    - Predictions are based on key factors like **Age, Job Level, Monthly Income, and Years at Company**.  
    - If your dataset contains additional columns (e.g., Department, Attrition), the tool will adapt and provide deeper insights.

    ### 📊 What You Can Do
    1. **Explore Workforce Data**  
       - View summary statistics and correlation heatmaps.  
       - Understand how variables (e.g., income, tenure) relate to attrition.  

    2. **Visualize Attrition Trends**  
       - Department-level attrition analysis.  
       - Age-group attrition patterns.  
       - Compare different groups to identify risk hotspots.  

    3. **Predict Employee Attrition**  
       - Test scenarios by entering employee details manually.  
       - Upload a full dataset and generate **employee-wise attrition probabilities**.  
       - Download predictions as CSV for further HR planning.  

    ### 🎯 Why Use This Tool?
    Employee turnover can be costly and disruptive.  
    This toolkit provides **data-driven insights** to help HR teams:
    - Identify employees at **high attrition risk**.  
    - Understand the **drivers of attrition**.  
    - Design **targeted retention strategies** (salary adjustments, career paths, training).  

    ---
    ⚠️ **Note:** This tool is a prototype for educational and research purposes.  
    For real-world use, predictions should be validated with larger datasets and tailored to your organization’s context.
    """)

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

        # Attrition by Department
        if "Department" in df.columns:
            dept_chart = df.groupby("Department")["Attrition"].mean().reset_index()
            st.bar_chart(dept_chart.set_index("Department"))
        else:
            st.info("No Department column found, skipping department analysis.")

        # Attrition by Age Group
        if "Age" in df.columns:
            df["AgeGroup"] = pd.cut(df["Age"], bins=[18,30,40,50,60],
                                    labels=["18-30","31-40","41-50","51-60"])
            age_chart = df.groupby("AgeGroup")["Attrition"].mean().reset_index()
            st.bar_chart(age_chart.set_index("AgeGroup"))

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

    # Bulk predictions if user uploaded data
    if uploaded_file:
        st.markdown("### Employee-Wise Attrition Predictions")
        df_pred = df.copy()
        df_pred["Attrition_Prob"] = model.predict_proba(df_pred[feature_cols])[:,1]
        df_pred["Attrition_Predicted"] = model.predict(df_pred[feature_cols])
        st.dataframe(df_pred.head(20))  # show first 20 employees

        # Download option
        csv = df_pred.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Full Predictions", csv,
                           "employee_attrition_predictions.csv", "text/csv")

st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("### 👨‍🏫 [Developed by Prof. Dinesh K.](https://linktr.ee/realdrdj)")
