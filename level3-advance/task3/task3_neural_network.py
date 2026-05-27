import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

print("=" * 70)
print("📌 ADVANCED LEVEL 3 TASK 3: FEED-FORWARD NEURAL NETWORK (SCI-KIT LEARN)")
print("=" * 70)

# ==========================================
# 🛠️ OBJECTIVE 1: LOAD AND PREPROCESS DATASET
# ==========================================
print("🔹 Step 1: Loading MNIST Dataset (this may take a moment over network)...")
# Fetching the official MNIST dataset directly from OpenML servers
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X, y = mnist.data, mnist.target

# Preprocessing: Normalize grayscale pixel values from 0-255 down to 0.0-1.0
X = X / 255.0

# Partition into 80% Train and 20% Validation Test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   -> Successfully partitioned data arrays.")
print(f"   -> Training frames: {X_train.shape[0]}")
print(f"   -> Testing frames:  {X_test.shape[0]}")


# ==========================================
# 🛠️ OBJECTIVES 2, 3 & 4: DESIGN, TUNE & TRAIN MODEL
# ==========================================
print("\n🔹 Step 2: Initializing Multi-Layer Perceptron (Neural Network)...")

# Hyperparameter Tuning Configuration:
# - hidden_layer_sizes=(128, 64) builds two hidden layers (Layer 1: 128 nodes, Layer 2: 64 nodes)
# - max_iter=5 trains it for 5 iteration epochs through the dataset
# - learning_rate_init=0.001 tunes our core optimization speed hyperparameter
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=5,
    random_state=42,
    verbose=True,
)

print("🚀 Starting model training via structural backpropagation...")
# Train the feed-forward weights
mlp.fit(X_train, y_train)


# ==========================================
# 🛠️ EVALUATION: OUT-OF-SAMPLE ACCURACY
# ==========================================
print("\n🔹 Step 3: Assessing Model on Unseen Validation Data...")
y_pred = mlp.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"   -> Final Out-of-Sample Accuracy: {test_acc * 100:.2f}%")


# ==========================================
# 🛠️ OBJECTIVE 5: VISUALIZE LOSS CURVE
# ==========================================
print("\n🔹 Step 4: Plotting Neural Network Loss Convergence Curve...")
plt.figure(figsize=(7, 5))
plt.plot(mlp.loss_curve_, color="crimson", lw=2, label="Training Loss")
plt.title("Neural Network Loss Convergence Curve", fontsize=11, fontweight="bold")
plt.xlabel("Iteration Epochs")
plt.ylabel("Loss Scale Value")
plt.legend()
plt.grid(True, alpha=0.4)

output_filename = "neural_network_training_curves.png"
plt.savefig(output_filename, dpi=300, bbox_inches="tight")
plt.close()

print(f"   ✅ Saved Convergence Plot Asset: '{output_filename}'")
print("\n" + "=" * 70)
print("🎉 FINAL INTERNSHIP MILESTONE SUCCESSFULLY COMPLETED!")
print("=" * 70)