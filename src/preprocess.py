import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path="data/HR_data.csv"):
    return pd.read_csv(path)

def preprocess_data(df):
    # Drop irrelevant columns if any
    df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], errors="ignore")

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    # Split features & target
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    return train_test_split(X, y, test_size=0.2, random_state=42)
