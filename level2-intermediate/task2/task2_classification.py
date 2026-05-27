import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 1. Configuration & Path Configuration
train_path = r"C:\Users\SWAMY\Desktop\project\level2-intermediate\task2\churn-bigml-80.csv"
test_path = r"C:\Users\SWAMY\Desktop\project\level2-intermediate\task2\churn-bigml-20.csv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    print("Error: Could not locate the churn train/test files. Verify paths.")
    exit()

# Load Train and Test partitions directly
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print("=" * 65)
print(" INTERMEDIATE LEVEL 2 TASK 2: CUSTOMER CHURN CLASSIFICATION")
print("=" * 65)


# ==========================================
#  OBJECTIVE 1: PREPROCESS THE DATA
# ==========================================
print("\n Step 1: Preprocessing Data (Encoding & Scaling)...")

# Define numerical features that require uniform scaling
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

# Standardize binary features into numeric 0/1 fields across both sets
le = LabelEncoder()
for df in [df_train, df_test]:
    df["International plan"] = le.fit_transform(
        df["International plan"].astype(str)
    )
    df["Voice mail plan"] = le.fit_transform(df["Voice mail plan"].astype(str))
    df["Churn"] = le.fit_transform(df["Churn"])

# Drop high-cardinality categorical attributes ('State', 'Area code') for pure feature calculation
X_train = df_train.drop(columns=["Churn", "State", "Area code"])
y_train = df_train["Churn"]

X_test = df_test.drop(columns=["Churn", "State", "Area code"])
y_test = df_test["Churn"]

# Apply StandardScaler to bring all feature variances to unit scale
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


# ==========================================
#  OBJECTIVES 2 & 4: TRAIN, EVALUATE, AND COMPARE MODELS
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
}

classification_metrics = {}

# Set up the plot for the ROC Curve comparison
plt.figure(figsize=(9, 7))

print("\n Training classification models and computing performance metrics...")
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Class predictions
    y_pred = model.predict(X_test)

    # Probability predictions needed for the ROC Curve calculation
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    classification_metrics[name] = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
    }

    # ==========================================
    #  OBJECTIVE 3: COMPUTE ROC CURVE METRICS
    # ==========================================
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot each model's line on the shared graph
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", lw=2)

# Print tabular comparison to the console
df_results = pd.DataFrame(classification_metrics).T
print("\n MODEL PERFORMANCE BENCHMARK:")
print(df_results)


# Complete ROC Curve plot formatting
plt.plot([0, 1], [0, 1], color="darkgrey", linestyle="--", label="Random Guess")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
plt.title(
    "Receiver Operating Characteristic (ROC) Curve Comparison",
    fontsize=13,
    fontweight="bold",
)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)

# Save chart asset
output_chart = "classification_roc_curve.png"
plt.savefig(output_chart, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n    Saved Classification Performance Plot: '{output_chart}'")
print("\n" + "=" * 65)
print(" Classification Milestone Complete!")
print("=" * 65)