import os
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# --- Download Crucial Language Assets ---
print("📥 Downloading required NLTK resources...")
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# 1. Configuration & Loading Data
file_path = r"C:\Users\SWAMY\Desktop\project\level3-advance\task2\3) Sentiment dataset.csv"

if not os.path.exists(file_path):
    print(f"Error: Could not locate dataset file at: {file_path}")
    exit()

df = pd.read_csv(file_path)

print("=" * 70)
print("📌 ADVANCED LEVEL 3 TASK 2: NLP TEXT CLASSIFICATION PIPELINE")
print("=" * 70)
print(f"Dataset Loaded. Total Records: {df.shape[0]}")

# Strip spaces from column names just in case
df.columns = df.columns.str.strip()


# ==========================================
# 🛠️ OBJECTIVE 1: PREPROCESS TEXT DATA
# ==========================================
print("\n🔹 Step 1: Preprocessing Text (Tokenization & Stopword Removal)...")

# Cache English stopwords (words like 'is', 'the', 'at' that don't add emotional meaning)
stop_words = set(stopwords.words('english'))

def clean_text(raw_text):
    if not isinstance(raw_text, str):
        return ""
    
    # Lowercase everything to make processing uniform
    lowercase_text = raw_text.lower()
    
    # Tokenization: Split the raw string sentence into individual word elements
    tokens = word_tokenize(lowercase_text)
    
    # Filtering: Keep words only if they are alphanumeric and not a stopword
    cleaned_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Re-join tokens back into a unified string sentence format
    return " ".join(cleaned_tokens)

# Apply text pipeline to the entire text column
df['Cleaned_Text'] = df['Text'].apply(clean_text)

# Split features from target labels
X = df['Cleaned_Text']
y = df['Sentiment'].str.strip() # Clean target text whitespace padding

# Partition data into 80% Training and 20% Validation Test arrays
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ==========================================
# 🛠️ OBJECTIVE 2: TF-IDF VECTORIZATION
# ==========================================
print("\n🔹 Step 2: Converting Text to Numerical Representations (TF-IDF)...")

# Initialize Term Frequency-Inverse Document Frequency Vectorizer
# This converts sentences into mathematical sparse matrix weights based on word importance
tfidf = TfidfVectorizer(max_features=2500)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"   -> Vocabulary shape matrix: {X_train_tfidf.shape}")


# ==========================================
# 🛠️ OBJECTIVE 3 & 4: MODEL TRAINING & METRIC EVALUATION
# ==========================================
print("\n🔹 Step 3: Training Logistic Regression Classification Model...")

# Train model on text features
classifier = LogisticRegression(random_state=42, max_iter=1000)
classifier.fit(X_train_tfidf, y_train)

# Generate predictions on the unseen test dataset
y_pred = classifier.predict(X_test_tfidf)

# Display deep metrics evaluation matrix
print("\n📋 MODEL METRICS MATRIX SUMMARY:")
print(f"Overall Accuracy Score: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred))

print("=" * 70)
print("🎉 NLP SENTIMENT CLASSIFICATION TASK COMPLETE!")
print("=" * 70)