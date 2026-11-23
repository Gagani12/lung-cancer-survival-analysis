 # IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

from pathlib import Path


#  CREATE OUTPUT FOLDER

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)



# LOAD DATASET

df = pd.read_csv("/content/lung_cancer_dataset.csv")

print("\n========== ORIGINAL DATASET ==========")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# CHECK DUPLICATES

print("\n========== DUPLICATE CHECK ==========")

print("Duplicate Rows:", df.duplicated().sum())

print(
    "Duplicate Patient IDs:",
    df["Patient_ID"].duplicated().sum()
)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove duplicate Patient_ID records if any exist
df = df.drop_duplicates(subset="Patient_ID", keep="first")

print("Shape After Duplicate Removal:", df.shape)

# CHECK MISSING VALUES


print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

print(missing_values)

print(
    "\nTotal Missing Values:",
    df.isnull().sum().sum()
)

missing_report = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isnull().sum().values,
    "Missing_Percentage": (
        df.isnull().mean().values * 100
    )
})

missing_report.to_csv(
    OUTPUT_DIR / "missing_values_report.csv",
    index=False
)

#  DATA TYPE CLEANING

print("\n========== DATA TYPE CLEANING ==========")

# Convert Diagnosis_Date to datetime format
df["Diagnosis_Date"] = pd.to_datetime(
    df["Diagnosis_Date"],
    errors="coerce"
)

# Clean text columns
categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
    )

    # Convert empty strings into missing values
    df[column] = df[column].replace(
        r"^\s*$",
        pd.NA,
        regex=True
    )
  print(df.dtypes)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

print(
    "Invalid Diagnosis Dates:",
    df["Diagnosis_Date"].isnull().sum()
)


# CHECK INVALID VALUES

print("\n========== INVALID VALUE CHECK ==========")

# Check numeric variables
numeric_columns = [
    "Age",
    "Cigarettes_Per_Day",
    "Years_Smoking",
    "BMI",
    "Tumor_Size_cm",
    "Survival_Months",
    "Diagnosis_Year"
]

print("\nNumeric Summary:")
print(df[numeric_columns].describe())

# Check logically invalid values
print("\nAge below 0:", (df["Age"] < 0).sum())

print(
    "Negative Cigarettes Per Day:",
    (df["Cigarettes_Per_Day"] < 0).sum()
)

print(
    "Negative Years Smoking:",
    (df["Years_Smoking"] < 0).sum()
)

print("BMI <= 0:", (df["BMI"] <= 0).sum())

print(
    "Tumor Size <= 0:",
    (df["Tumor_Size_cm"] <= 0).sum()
)

print(
    "Survival Months <= 0:",
    (df["Survival_Months"] <= 0).sum()
)

# Check categorical values
print("\nGender Categories:")
print(df["Gender"].value_counts())

print("\nCancer Stage Categories:")
print(df["Cancer_Stage"].value_counts())

print("\nSurvival Status Categories:")
print(df["Survived"].value_counts())

# SAVE CLEANED DATASET

df.to_csv(
    OUTPUT_DIR / "lung_cancer_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

print("Final Dataset Shape:", df.shape)
