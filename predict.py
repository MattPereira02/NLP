import joblib
import re
import numpy as np
import pandas as pd
from preprocessing import preprocess_new_data

tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")
scaler = joblib.load("scaler.joblib")
model = joblib.load("random_forest_spam_classifier.joblib")

def predict_spam(cleaned_message):
    """Classifies a new message as spam or not spam."""
    feature_names = tfidf_vectorizer.get_feature_names_out()
    transformed_message = tfidf_vectorizer.transform([cleaned_message])
    message_df = pd.DataFrame(transformed_message.toarray(), columns=feature_names)
    
    # Get the columns that the scaler is expecting
    scaler_columns = scaler.feature_names_in_
    
    # Reindex the message_df to match the scaler's columns
    message_df = message_df.reindex(columns=scaler_columns, fill_value=0)
    
    scaled_message = scaler.transform(message_df)
    prediction = model.predict(scaled_message)  # Predict spam (1) or legit (0)
    return "Spam" if prediction[0] == 1 else "Legitimate"

# User input for testing
if __name__ == "__main__":
    message = input("Enter a message to classify: ")
    cleaned_message = preprocess_new_data(message)
    result = predict_spam(cleaned_message)
    print(f"Prediction: {result}")

def feature_checker(cleaned_message):
    print(cleaned_message)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return feature_names