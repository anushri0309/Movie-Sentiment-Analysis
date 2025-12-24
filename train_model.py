# train_model.py
import os
import re
import numpy as np
import pandas as pd
import kagglehub
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


def download_and_load_data(limit_rows=5000):
    # Download dataset from Kaggle
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    print("Dataset path:", path)

    # Load CSV
    csv_path = os.path.join(path, "IMDB Dataset.csv")
    data = pd.read_csv(csv_path)
    if limit_rows is not None:
        data = data.iloc[:limit_rows].copy()

    print("Dataset loaded! Shape:", data.shape)
    print("\nSample:")
    print(data.head())
    return data


def preprocess_reviews(data):
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    corpus = []
    for i in range(len(data)):
        review = re.sub("[^a-zA-Z]", " ", data["review"][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if word not in stop_words]
        review = " ".join(review)
        corpus.append(review)

    print(f"\nProcessed {len(corpus)} reviews")
    return corpus


def main():
    # 1. Load data
    data = download_and_load_data(limit_rows=5000)

    # 2. Preprocess
    corpus = preprocess_reviews(data)

    # 3. Labels
    y = data["sentiment"].map({"positive": 1, "negative": 0}).astype(int).values.ravel()
    print("y shape:", y.shape)
    print("Labels sample:", y[:10])
    print("Unique labels:", np.unique(y))

    # 4. TF-IDF
    cv = TfidfVectorizer(max_features=5000)
    X = cv.fit_transform(corpus).toarray()
    print("X shape:", X.shape)

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train shapes:", X_train.shape, y_train.shape)
    print("Test shapes:", X_test.shape, y_test.shape)

    # 6. Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nFINAL ACCURACY: {accuracy:.3f}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"y_test counts (neg, pos): {np.bincount(y_test)}")
    print(f"y_pred counts (neg, pos): {np.bincount(y_pred)}")

    # 8. Save model + vectorizer
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "model.joblib")
    vect_path = os.path.join("models", "vectorizer.joblib")

    joblib.dump(model, model_path)
    joblib.dump(cv, vect_path)

    print("\n✅ Model and vectorizer saved!")
    print(f"Model: {model_path}")
    print(f"Vectorizer: {vect_path}")


if __name__ == "__main__":
    main()
