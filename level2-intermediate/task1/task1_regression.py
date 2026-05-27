#PREDECTIVE MODELING (REGRESSION)
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# 1. Configuration & Path Configuration
# Pointing directly to the space-separated text file in your directory
file_path = (
    r"C:\Users\SWAMY\Desktop\project\level2-intermediate\task1\4) house Prediction Data Set.csv"
)

if not os.path.exists(file_path):
    print(f"Error: Could not locate dataset file at: {file_path}")
    exit()

# 2. Loading and Mapping Column Structures
# Using a regex-based whitespace delimiter sep=r'\s+' because columns are space-separated
boston_cols = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]
df_house = pd.read_csv(file_path, sep=r"\s+", header=None, names=boston_cols)

print("=" * 65)
print(" INTERMEDIATE LEVEL 2 TASK 1: HOUSING REGRESSION ANALYSIS")
print("=" * 65)
print(f"Dataset Loaded Successfully. Total Rows: {df_house.shape[0]}")

# Split Features (X) from the continuous numerical target vector (y)
X = df_house.drop(columns=["MEDV"])
y = df_house["MEDV"]


# ==========================================
#  OBJECTIVE 1: SPLIT TRAIN AND TEST SETS
# ==========================================
# Allocating 80% of the observations to training and 20% to validation testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"-> Training Data: {X_train.shape[0]} rows")
print(f"-> Testing Data:  {X_test.shape[0]} rows")


# ==========================================
#  OBJECTIVES 2 & 4: MODELING AND EXPERIMENTATION
# ==========================================
# Defining algorithms to contrast a baseline linear trend against non-linear architectures
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
}

# Dictionary to capture evaluation scores
performance_metrics = {}

print("\ Training models and calculating evaluation scores...")
for name, model in models.items():
    # Fit model weights using training subsets
    model.fit(X_train, y_train)

    # Generate predictions on the validation test frame
    y_pred = model.predict(X_test)

    # ==========================================
    # OBJECTIVE 3: EVALUATE PERFORMANCE METRICS
    # ==========================================
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store computed values
    performance_metrics[name] = {"MSE": round(mse, 4), "R2": round(r2, 4)}

# Display results tabularly in console
df_results = pd.DataFrame(performance_metrics).T
print("\n MODEL METRICS COMPARISON TABLE:")
print(df_results)


# ==========================================
# 🛠️ OBJECTIVE 5: PERFORMANCE VISUALIZATION
# ==========================================
print("\nGenerating performance evaluation bar charts...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Chart A: Mean Squared Error (Lower is better)
df_results["MSE"].plot(
    kind="bar", ax=axes[0], color=["#ff7f7f", "#ffbf7f", "#7fbf7f"], edgecolor="black"
)
axes[0].set_title("Mean Squared Error (Lower = Better)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("MSE Scale")
axes[0].set_xticklabels(df_results.index, rotation=0)

# Chart B: R-squared Score (Higher is better)
df_results["R2"].plot(
    kind="bar", ax=axes[1], color=["#ff7f7f", "#ffbf7f", "#7fbf7f"], edgecolor="black"
)
axes[1].set_title("R-squared Score (Higher = Better)", fontsize=11, fontweight="bold")
axes[1].set_ylabel("R2 Value Range (0 to 1)")
axes[1].set_xticklabels(df_results.index, rotation=0)

plt.suptitle(
    "Regression Model Benchmark Performance Summary", fontsize=13, fontweight="bold", y=0.98
)
plt.tight_layout()

# Save figure directly into the path directory
output_image = "regression_model_comparison.png"
plt.savefig(output_image, dpi=300, bbox_inches="tight")
plt.close()

print(f"    Saved Comparison Plot Asset: '{output_image}'")
print("\n" + "=" * 65)
print(" Model Benchmarking Complete! Ready for Evaluation Notes.")
print("=" * 65)