# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from config import COLUMNS
import joblib
import re

tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")

def clean_text(X):
    """Clean the text data in X by removing spaces and special characters."""
    X = X.apply(lambda x: ''.join(e for e in str(x) if e.isalnum()) if isinstance(x, str) else x)
    return X

def preprocess_data(data):
    """Preprocess the dataset by scaling the features and preparing the target variable."""
    
    # Extract features and target
    X = data.iloc[:, :-1]  # All columns except the last one (which is the label)
    y = data['label']  # The label column

    X= X.apply(lambda x: clean_text(x))

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    # Save the TF-IDF vectorizer for later use
    joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.joblib")
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler for later use
    joblib.dump(scaler, "scaler.joblib")

    return X_scaled, y

def preprocess_new_data(text):
    """Preprocess new data using the same transformation as the training data."""
    cleaned_message = ''.join(e for e in str(text) if e.isalnum())
    return cleaned_message