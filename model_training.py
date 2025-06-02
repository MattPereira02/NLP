from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from config import RANDOM_STATE
import joblib

def train_random_forest(X_train, y_train):
    """Train Random Forest model on the training data."""
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name="Random Forest"):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Displays metrics
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Displays classification report
    report = classification_report(y_test, y_pred)
    print(f"\nClassification Report for {model_name}:\n", report)

    # Displays confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Spam"], yticklabels=["Legit", "Spam"])
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def cross_validate_model(model, X_train, y_train):
    # Perform 5-fold cross-validation and compute the mean accuracy
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    mean_cv_score = cv_scores.mean()
    
    print(f"\nCross-validation Accuracy for Random Forest: {mean_cv_score:.4f}")

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train the Random Forest model, perform cross-validation, and evaluate it on the test set."""
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)

    # Cross-validation
    cross_validate_model(rf_model, X_train, y_train)
    
    # Evaluate on test data
    evaluate_model(rf_model, X_test, y_test)

        #Save the trained model to a file
    joblib.dump(rf_model, "random_forest_spam_classifier.joblib")
