import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import load_data, preprocess_for_prediction
from model import train_model, evaluate_model
from dashboard import dashboard

st.set_page_config(page_title="HR Analytics Toolkit", layout="wide")
st.title("🧑‍💼 Pythonic HR Analytics Toolkit")

# ----------------- Load dataset -----------------
df = load_data()

# Define consistent features
feature_cols = ["Age", "JobLevel", "MonthlyIncome", "YearsAtCompany"]

# Train model once at startup
X_train, X_test, y_train, y_test = preprocess_for_prediction(df)
model = train_model(X_train[feature_cols], y_train)
acc, report = evaluate_model(model, X_test[feature_cols], y_test)

# ----------------- Sidebar Menu -----------------
menu = st.sidebar.radio(
    "Choose a section",
    ["Overview", "EDA", "Attrition Dashboard", "Attrition Prediction"]
)

# ----------------- Overview -----------------
if menu == "Overview":
    st.subheader("Project Overview")
    st.markdown("""
    This toolkit demonstrates **practical HR analytics**:
    - Explore workforce data (EDA).
    - Visualize attrition patterns.
    - Predict employee attrition probability.
    """)
    st.write("### Model Accuracy")
    st.write(f"Model Accuracy (on test set): **{acc:.2%}**")

# ----------------- EDA -----------------
elif menu == "EDA":
    st.subheader("Exploratory Data Analysis")
    st.write("### Dataset Preview")
    st.write(df.head())

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    if len(numeric_cols) > 1:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=True, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for correlation heatmap.")

# ----------------- Dashboard -----------------
elif menu == "Attrition Dashboard":
    if "Attrition" not in df.columns:
        st.error("Attrition column missing in dataset.")
    else:
        st.subheader("Attrition by Age Group")
        df["AgeGroup"] = pd.cut(
            df["Age"], bins=[18,30,40,50,60],
            labels=["18-30","31-40","41-50","51-60"]
        )
        fig, ax = plt.subplots()
        df.groupby("AgeGroup")["Attrition"].value_counts(normalize=True)\
          .unstack().plot(kind="bar", stacked=True, ax=ax)
        st.pyplot(fig)

# ----------------- Prediction -----------------
elif menu == "Attrition Prediction":
    st.subheader("Predict Employee Attrition")

    # Collect input features
    age = st.number_input("Age", 18, 60, 30)
    job_level = st.slider("Job Level", 1, 5, 2)
    monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
    years_at_company = st.slider("Years at Company", 0, 40, 5)

    # Build DataFrame with same feature names
    input_data = pd.DataFrame(
        [[age, job_level, monthly_income, years_at_company]],
        columns=feature_cols
    )

    # Predict
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.write(f"### Prediction: {'Attrition' if pred==1 else 'No Attrition'}")
    st.write(f"Probability of leaving: **{prob:.2%}**")
