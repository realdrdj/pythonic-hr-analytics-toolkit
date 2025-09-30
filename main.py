import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="HR Analytics Toolkit", layout="wide")
st.title("🧑‍💼 Pythonic HR Analytics Toolkit")

# ----------------- Load Data -----------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/HR_data.csv")
    except:
        # Fallback demo dataset if file not found
        df = pd.DataFrame({
            "Age": [25, 30, 45, 40, 28, 35],
            "JobLevel": [1, 2, 3, 2, 1, 4],
            "MonthlyIncome": [3000, 5000, 12000, 8000, 4000, 15000],
            "YearsAtCompany": [1, 5, 10, 8, 2, 12],
            "Attrition": [1, 0, 0, 1, 0, 0]
        })
    return df

df = load_data()

# ----------------- Preprocess -----------------
def preprocess_for_prediction(df):
    X = df[["Age", "JobLevel", "MonthlyIncome", "YearsAtCompany"]]
    y = df["Attrition"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

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

# ----------------- Train at Startup -----------------
feature_cols = ["Age", "JobLevel", "MonthlyIncome", "YearsAtCompany"]
X_train, X_test, y_train, y_test = preprocess_for_prediction(df)
model = train_model(X_train, y_train)
acc, report = evaluate_model(model, X_test, y_test)

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
