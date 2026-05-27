#EXPLORATORY DATA ANALYSIS (EDA)
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load the dataset
file_path = r"C:\Users\SWAMY\Desktop\project\level1-basic\task3\1) iris.csv"

if not os.path.exists(file_path):
    print(
        f"Error: Could not find '{file_path}'. Please ensure it is in this folder."
    )
    exit()

df = pd.read_csv(file_path)

print("=" * 50)
print(" EXPLORATORY DATA ANALYSIS REPORT: IRIS DATASET")
print("=" * 50)

# --- OBJECTIVE 1: SUMMARY STATISTICS ---
print("\n 1. Dataset Structure & Summary Statistics:")
print(f"Total Rows (Samples): {df.shape[0]}")
print(f"Total Columns (Features): {df.shape[1]}")
print(f"Species Breakdown:\n{df['species'].value_counts()}\n")

# Calculate comprehensive summary statistics
stats = df.describe().T
# Add variance explicitly since describe() doesn't include it automatically
stats["variance"] = df.select_dtypes(include=["float64", "int64"]).var()
print(stats[["mean", "50%", "std", "variance", "min", "max"]])
# Note: '50%' represents the Median value

# --- OBJECTIVE 2: CORRELATION MATRIX ---
print("\n 2. Correlation Matrix (Numerical Features):")
# Drop the categorical 'species' column to calculate correlations
numerical_df = df.drop(columns=["species"])
correlation_matrix = numerical_df.corr()
print(correlation_matrix)


# --- OBJECTIVE 3: DATA VISUALIZATIONS ---
print("\n 3. Generating Visualization Charts...")

# Set standard styles for clean visual reporting
sns.set_theme(style="whitegrid")

# Plot 1: Histograms (Feature Distributions)
# Automatically creates grid layout for features without explicit .figure()
axes = numerical_df.hist(figsize=(10, 8), bins=15, color="skyblue", edgecolor="black")
plt.suptitle("Feature Distributions (Histograms)", fontsize=14, y=0.95)
plt.savefig("iris_histograms.png", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved: iris_histograms.png")

# Plot 2: Box Plots (Outlier Detection & Feature Spread)
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=numerical_df, ax=ax, palette="Set2")
ax.set_title("Feature Spreads & Outlier Detection (Box Plots)", fontsize=14)
ax.set_ylabel("Measurements (cm)")
plt.savefig("iris_boxplots.png", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved: iris_boxplots.png")

# Plot 3: Scatter Plot (Relationship between features)
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="petal_length",
    y="petal_width",
    hue="species",
    palette="deep",
    s=70,
    ax=ax,
)
ax.set_title("Petal Length vs Petal Width Relationship", fontsize=14)
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
plt.savefig("iris_scatterplot.png", dpi=300, bbox_inches="tight")
plt.close()
print("    Saved: iris_scatterplot.png")

print("\n" + "=" * 50)
print(" EDA Processing Complete! Check your folder for images.")
print("=" * 50)