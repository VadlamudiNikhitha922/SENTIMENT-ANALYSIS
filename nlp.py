import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset (example dataset: IMDb reviews)
data = "db.csv"
data = pd.read_csv(data)

# Display the first few rows of the dataset
print("Dataset sample:")
print(data.head())

# Select relevant columns (assuming 'text' for reviews and 'sentiment' for labels)
reviews = data['text']
labels = data['sentiment']

# Preprocess labels (convert positive/negative to binary: 1 and 0)
labels = labels.map({'positive': 1, 'negative': 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# Model Evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Example prediction
sample_review = ["The product was excellent and arrived on time."]
sample_tfidf = vectorizer.transform(sample_review)
sentiment = model.predict(sample_tfidf)
sentiment_label = "positive" if sentiment[0] == 1 else "negative"
print(f"Sample Review Sentiment: {sentiment_label}")
