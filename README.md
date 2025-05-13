# Spam Detection System Using Random Forest

## Project Overview
This project implements a machine learning-based spam detection system using the **Spambase dataset** and a **Random Forest Classifier**. The system can classify incoming text messages as either **Spam** or **Legitimate** based on trained features.

## Project Structure
- `main.py`: Runs the pipeline.
- `model_training.py`: Trains and saves the model.
- `preprocessing.py`: Prepares the dataset and new inputs.
- `predict.py`: Uses the trained model to classify new messages.
- `config.py`: Stores file paths and constants.
- `models/`: Folder containing the trained `.joblib` files.
- `data/spambase.data`: Dataset used for training/testing.

## Model Summary
- **Model**: Random Forest Classifier
- **Dataset**: UCI Spambase (4,601 messages, 57 features)
- **Training/Testing Split**: 80/20
- **Preprocessing**: TF-IDF Vectorization, Standard Scaling
- **Binary Output**: Legitimate (0) or Spam (1)

### Performance Metrics

**Classification Report for Random Forest:**

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| 0 (Legitimate)| 0.94      | 0.98   | 0.96     | 531     |
| 1 (Spam)      | 0.98      | 0.92   | 0.95     | 390     |

**Accuracy:** 0.96  
**Total Support:** 921

**Averaged Scores:**

| Average Type   | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Macro Avg      | 0.96      | 0.95   | 0.95     | 921     |
| Weighted Avg   | 0.96      | 0.96   | 0.96     | 921     |

## How to Use
1. Clone the repo.
2. Ensure you have Python 3.8+ and required packages installed.
3. Run the training pipeline:
   ```bash
   python main.py
