import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_PATH, COLUMNS, TEST_SIZE, RANDOM_STATE
from preprocessing import preprocess_data
from model_training import train_and_evaluate

# Load the dataset using the path and columns from config.py
data = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

# Preprocess the data (scaling features and extracting labels)
X, y = preprocess_data(data)

# Split the dataset into training and test sets using values from config.py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Train and evaluate the Random Forest model
train_and_evaluate(X_train, X_test, y_train, y_test)