import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def dashboard(df):
    st.title("Pythonic HR Analytics Dashboard")

    st.subheader("Attrition Distribution")
    fig, ax = plt.subplots()
    df["Attrition"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("Average Monthly Income by Job Role")
    fig, ax = plt.subplots()
    df.groupby("JobRole")["MonthlyIncome"].mean().sort_values().plot(kind="barh", ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    st.write(df.corr())
