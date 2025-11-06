# ============================================================
# PROJECT: Sentiment Analysis System using Naive Bayes Algorithm
# AUTHOR : [Your Name] (Reg No: [Your Register Number])
# ============================================================
# Run this file directly in VS Code using Python extension
# Make sure sentiment_dataset.csv is in the same folder
# ============================================================

# Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
import re
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1: LOAD THE DATASET
# ============================================================
data = pd.read_csv("sentiment_dataset.csv")  # Ensure file is in the same folder
print("✅ Dataset Loaded Successfully!")
print("\nDataset Shape:", data.shape)
print("\nFirst 5 Records:\n", data.head())
print("\nSentiment Distribution:\n", data['sentiment'].value_counts())

# ============================================================
# STEP 2: TEXT PREPROCESSING
# ============================================================
def preprocess_text(text):
    """Clean and preprocess text data"""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

data['cleaned_text'] = data['review'].apply(preprocess_text)
print("\n✅ Text Preprocessing Completed!")

# ============================================================
# STEP 3: FEATURE EXTRACTION
# ============================================================
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['sentiment']

print("\n✅ Feature Extraction Completed!")
print("Feature Matrix Shape:", X.shape)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# STEP 4: MODEL TRAINING
# ============================================================
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
print("\n✅ Model Training Completed Successfully!")

# ============================================================
# STEP 5: EVALUATION
# ============================================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print("=" * 50)
print("Accuracy :", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ============================================================
# STEP 6: SAVE THE MODEL AND VECTORIZER
# ============================================================
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\n💾 Model saved as 'sentiment_model.pkl'")
print("💾 Vectorizer saved as 'tfidf_vectorizer.pkl'")

# ============================================================
# STEP 7: PREDICT SENTIMENT FROM USER INPUT
# ============================================================
print("\n" + "=" * 50)
print("💬 Sentiment Analysis System (User Mode)")
print("=" * 50)

while True:
    user_review = input("\nEnter a review (or 'quit' to exit): ")
    
    if user_review.lower() == 'quit':
        print("👋 Thank you for using the Sentiment Analysis System!")
        break
    
    # Preprocess and vectorize user input
    cleaned_review = preprocess_text(user_review)
    review_vector = vectorizer.transform([cleaned_review])
    
    # Predict sentiment
    prediction = model.predict(review_vector)[0]
    probability = model.predict_proba(review_vector)[0]
    
    print("\n" + "=" * 50)
    print("🎯 Predicted Sentiment:", prediction.upper())
    print(f"📈 Confidence: {max(probability) * 100:.2f}%")
    print("=" * 50)

# ============================================================
# STEP 8: VERIFY LOADED MODEL
# ============================================================
loaded_model = joblib.load("sentiment_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")
test_pred = loaded_model.predict(X_test)
print("\nReloaded Model Accuracy:", round(accuracy_score(y_test, test_pred) * 100, 2), "%")
print("\n✅ Program Executed Successfully!")