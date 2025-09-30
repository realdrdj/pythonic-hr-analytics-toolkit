from src.preprocess import load_data, preprocess_data
from src.model import train_model, evaluate_model
from src.dashboard import dashboard
import streamlit as st

def run_training():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    acc, report = evaluate_model(model, X_test, y_test)

    print("Model Accuracy:", acc)
    print(report)

def run_dashboard():
    df = load_data()
    dashboard(df)

if __name__ == "__main__":
    choice = input("Type 'train' to run ML model or 'dash' to launch dashboard: ")
    if choice == "train":
        run_training()
    else:
        import os
        os.system("streamlit run main.py")
