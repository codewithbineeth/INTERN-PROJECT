import pandas as pd
import numpy as np
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Ensure resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

file_path = r"C:\Users\SWAMY\Desktop\project\level3-advance\task2\3) Sentiment dataset.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# --- THE FIX: GROUP LABELS INTO THREE CORE SENTIMENTS ---
def group_sentiment(label):
    label = str(label).strip().lower()
    # List words that indicate negative sentiments
    negatives = ['negative', 'anger', 'despair', 'bad', 'disgust', 'fear', 'frustrated', 'frustration', 'sadness', 'sorrow', 'bitter', 'bitterness', 'heartbreak', 'hate', 'desolation', 'betrayal']
    # List words that indicate positive sentiments
    positives = ['positive', 'happy', 'joy', 'joyfulreunion', 'excitement', 'elation', 'proud', 'grateful', 'inspiration', 'inspired', 'euphoria', 'thrill', 'vibrancy', 'zest', 'affection', 'admiration', 'contentment', 'serenity', 'fulfillment', 'spark', 'acceptance']
    
    if any(neg in label for neg in negatives):
        return 'Negative'
    elif any(pos in label for pos in positives):
        return 'Positive'
    else:
        return 'Neutral'

# Map messy classes into clean buckets
df['Clean_Sentiment'] = df['Sentiment'].apply(group_sentiment)

print(f"Grouped breakdown:\n{df['Clean_Sentiment'].value_counts()}")

# --- Standard Text Preprocessing Pipeline ---
stop_words = set(stopwords.words('english'))
def clean_text(raw_text):
    if not isinstance(raw_text, str): return ""
    tokens = word_tokenize(raw_text.lower())
    return " ".join([w for w in tokens if w.isalnum() and w not in stop_words])

df['CleanED_Text'] = df['Text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df['CleanED_Text'], df['Clean_Sentiment'], test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=1500)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Re-train on Grouped Targets
classifier = LogisticRegression(random_state=42, max_iter=1000)
classifier.fit(X_train_tfidf, y_train)
y_pred = classifier.predict(X_test_tfidf)

print("\n" + "="*50)
print(f"🎉 OPTIMIZED ACCURACY SCORE: {accuracy_score(y_test, y_pred):.4f}")
print("="*50)
print(classification_report(y_test, y_pred))