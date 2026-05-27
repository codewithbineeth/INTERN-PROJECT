#DATA CLEANING AND PREPROCESSING
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Configuration & Loading Data
# Adjust the path to match wherever your dataset folder sits
file_path = r"C:\Users\SWAMY\Desktop\project\level1-basic\task2\churn-bigml-80.csv"

if not os.path.exists(file_path):
    print(f"Error: Could not find '{file_path}'. Please verify the path.")
    exit()

df = pd.read_csv(file_path)

print("=" * 60)
print(" STARTING DATA CLEANING & PREPROCESSING PIPELINE")
print("=" * 60)

# --- SIMULATING RAW EMPTY DATA FOR DEMONSTRATION ---
np.random.seed(42)
df_raw = df.copy()
# Intentionally inject missing values (NaNs) into 5% of numerical columns
mask_num1 = np.random.rand(len(df_raw)) < 0.05
mask_num2 = np.random.rand(len(df_raw)) < 0.05
mask_cat = np.random.rand(len(df_raw)) < 0.03

df_raw.loc[mask_num1, "Account length"] = np.nan
df_raw.loc[mask_num2, "Total day minutes"] = np.nan
df_raw.loc[mask_cat, "International plan"] = np.nan


# ==========================================
#  OBJECTIVE 1: HANDLE MISSING DATA (IMPUTATION)
# ==========================================
print(" Step 1: Handling Missing Data...")
print("Missing counts before treatment:")
print(df_raw.isnull().sum()[df_raw.isnull().sum() > 0])

# Using SimpleImputer to fill numerical columns with their Median value
num_imputer = SimpleImputer(strategy="median")
df_raw[["Account length", "Total day minutes"]] = num_imputer.fit_transform(
    df_raw[["Account length", "Total day minutes"]]
)

# Using SimpleImputer to fill categorical columns with the Most Frequent value (Mode)
cat_imputer = SimpleImputer(strategy="most_frequent")
df_raw["International plan"] = cat_imputer.fit_transform(
    df_raw[["International plan"]]
).ravel()

print(f"Total missing values remaining: {df_raw.isnull().sum().sum()}")



#  OBJECTIVE 2: DETECT AND REMOVE OUTLIERS (IQR METHOD)

print("\n Step 2: Detecting & Removing Outliers...")
print(f"Total rows before outlier filter: {len(df_raw)}")

# List of numerical columns tracking core metrics
numerical_cols = [
    "Account length",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total day charge",
    "Total eve minutes",
    "Total eve calls",
    "Total eve charge",
    "Total night minutes",
    "Total night calls",
    "Total night charge",
    "Total intl minutes",
    "Total intl calls",
    "Total intl charge",
    "Customer service calls",
]

# We will apply the standard Interquartile Range (IQR) filter on key active usage features
for col in ["Total day minutes", "Total eve minutes", "Total night minutes"]:
    Q1 = df_raw[col].quantile(0.25)
    Q3 = df_raw[col].quantile(0.75)
    IQR = Q3 - Q1

    # Define boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out values outside the bounds
    df_raw = df_raw[(df_raw[col] >= lower_bound) & (df_raw[col] <= upper_bound)]

print(f"Total rows after outlier removal: {len(df_raw)}")



#  OBJECTIVE 3: ENCODE CATEGORICAL VARIABLES

print("\n Step 3: Encoding Categorical Variables into Numeric format...")

# Use LabelEncoder for Binary (Yes/No or True/False) Categorical Features
le = LabelEncoder()
df_raw["International plan"] = le.fit_transform(
    df_raw["International plan"].astype(str)
)
df_raw["Voice mail plan"] = le.fit_transform(
    df_raw["Voice mail plan"].astype(str)
)
df_raw["Churn"] = le.fit_transform(df_raw["Churn"])

# Use One-Hot Encoding for Multi-class Feature ('State') using pd.get_dummies()
# We use drop_first=True to protect against the dummy variable trap in ML
df_raw = pd.get_dummies(df_raw, columns=["State"], drop_first=True, dtype=int)

print(
    f"Categorical encoding completed. Data converted successfully to numeric fields."
)



#  OBJECTIVE 4: NORMALIZE / STANDARDIZE NUMERICAL DATA

print("\n🔹 Step 4: Standardizing Continuous Numerical Fields...")

# Initialize StandardScaler (Centers data to mean = 0, variance = 1)
scaler = StandardScaler()
df_raw[numerical_cols] = scaler.fit_transform(df_raw[numerical_cols])

print("Sample values post-standardization (Mean ≈ 0, Std ≈ 1):")
print(df_raw[numerical_cols].head(2).T)



#  SAVE RESULTS

output_filename = "cleaned_churn_data.csv"
df_raw.to_csv(output_filename, index=False)

print("\n" + "=" * 60)
print(f" SUCCESS: Pipeline complete! Cleaned shape: {df_raw.shape}")
print(f"Processed file safely saved as: '{output_filename}'")
print("=" * 60)