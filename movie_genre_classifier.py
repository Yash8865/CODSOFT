# movie_genre_classifier.py

"""
Multi-label Movie Genre Classification using TF-IDF + Logistic Regression

Dataset: IMDb_5000 movie dataset (auto-downloaded)
Model: TF-IDF + Logistic Regression (One-Vs-Rest)
"""

import pandas as pd
import numpy as np
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def download_dataset():
    # Public IMDb dataset (~5000 movies)
    url = "https://raw.githubusercontent.com/datasets/imdb/master/data/movies.csv"
    response = requests.get(url).content
    df = pd.read_csv(io.BytesIO(response))
    return df

def preprocess(df):
    df = df.dropna(subset=['Genre', 'Title'])
    # Split genre string into list
    df['Genre'] = df['Genre'].apply(lambda x: x.split(','))
    return df

def prepare_data(df):
    X = df['Title']  # Using Title as input feature (can be replaced with Description if available)
    y = df['Genre']

    # Convert labels to multi-hot encoding
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)

    return X, y, mlb

def main():
    print("Downloading & loading dataset...")
    df = download_dataset()

    print("Preprocessing data...")
    df = preprocess(df)

    print("Preparing data...")
    X, y, mlb = prepare_data(df)

    print("Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_tfidf = tfidf.fit_transform(X)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    print("Training One-Vs-Rest Logistic Regression model...")
    clf = OneVsRestClassifier(LogisticRegression(max_iter=300))
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))

    # Optional accuracy (for multi-label use micro/macro)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy (micro-averaged): {acc:.4f}")

    # Save model + vectorizer + label binarizer
    print("Saving model to disk...")
    joblib.dump(clf, "genre_classifier.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    joblib.dump(mlb, "label_binarizer.pkl")
    print("Saved as: genre_classifier.pkl, tfidf_vectorizer.pkl, label_binarizer.pkl")

if __name__ == "__main__":
    main()
