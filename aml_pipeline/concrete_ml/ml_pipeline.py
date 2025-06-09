import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
import os
from concrete.ml.sklearn import XGBClassifier
import onnx
from concrete.ml.onnx import convert_onnx_to_numpy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
MODEL_PATH = "model.joblib"
SCALER_PATH = "scaler.joblib"
ONNX_PATH = "model.onnx"

def load_and_preprocess_data(file_path):
    """Load and preprocess the data."""
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_and_save_model(X_train, y_train, fhe_mode='disable'):
    """Train the model and save it."""
    logger.info(f"Training model with FHE mode: {fhe_mode}")
    
    # Initialize and train the model
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        n_bits=8,
        fhe_mode=fhe_mode
    )
    
    clf.fit(X_train, y_train)
    
    # Save model and scaler only if FHE is disabled
    if fhe_mode == 'disable':
        logger.info(f"Saving model to {MODEL_PATH}")
        joblib.dump(clf, MODEL_PATH)
        logger.info(f"Saving scaler to {SCALER_PATH}")
        joblib.dump(scaler, SCALER_PATH)
    else:
        logger.info("Skipping model save: FHE mode enabled (model not serializable)")
    
    return clf

def evaluate_model(clf, X_test, y_test):
    """Evaluate the model performance."""
    logger.info("Evaluating model performance")
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    logger.info(f"Model metrics: {metrics}")
    return metrics

def convert_to_onnx(clf, X_train):
    """Convert the model to ONNX format."""
    logger.info("Converting model to ONNX format")
    
    # Export to ONNX
    onnx_model = clf.export_model_to_onnx(X_train)
    
    # Save ONNX model
    onnx.save(onnx_model, ONNX_PATH)
    logger.info(f"ONNX model saved to {ONNX_PATH}")
    
    return onnx_model

def main():
    """Main function to run the pipeline."""
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("data.csv")
    
    # Train model with FHE disabled
    clf = train_and_save_model(X_train, y_train, fhe_mode='disable')
    
    # Evaluate model
    metrics = evaluate_model(clf, X_test, y_test)
    
    # Convert to ONNX
    onnx_model = convert_to_onnx(clf, X_train)
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main() 