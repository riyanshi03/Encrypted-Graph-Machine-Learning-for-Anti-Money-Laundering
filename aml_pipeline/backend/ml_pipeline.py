import pandas as pd
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from concrete.ml.sklearn.xgb import XGBClassifier
from gfp import gfp_enrichment
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths - using absolute paths in the container
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Data file paths
DATA_PATH = os.path.join(DATA_DIR, 'HI-Small_Sampled_5491.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

FEATURES = [
    'SourceAccountId', 'TargetAccountId', 'Timestamp',
    'Amount Sent', 'Sent Currency', 'Amount Received',
    'Receiving Currency', 'Payment Format'
]
TARGET = 'Is Laundering'

def validate_input_data(input_dict):
    """Validate input data for prediction"""
    # Check if all required features are present
    missing_features = [f for f in FEATURES if f not in input_dict]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Validate data types and ranges
    for feature in FEATURES:
        value = input_dict[feature]
        if feature in ['Amount Sent', 'Amount Received']:
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{feature} must be a non-negative number")
        elif feature == 'Timestamp':
            try:
                pd.to_datetime(value)
            except:
                raise ValueError(f"{feature} must be a valid datetime")
        elif feature in ['SourceAccountId', 'TargetAccountId']:
            if not isinstance(value, (int, str)):
                raise ValueError(f"{feature} must be an integer or string")
        elif feature in ['Sent Currency', 'Receiving Currency', 'Payment Format']:
            if not isinstance(value, str):
                raise ValueError(f"{feature} must be a string")

def ensure_directories():
    """Ensure required directories exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Model directory: {MODEL_DIR}")

def check_data_file():
    """Check if data file exists and is accessible"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}. "
            "Please ensure the data file is present in the data directory."
        )
    try:
        # Try to read the first few lines to verify file is accessible
        pd.read_csv(DATA_PATH, nrows=1)
    except Exception as e:
        raise IOError(f"Error reading data file: {str(e)}")

def add_edge_id(df):
    """Add EdgeID column to dataframe"""
    df['EdgeID'] = range(len(df))
    return df

def train_and_save_model(gfp=False, fhe_mode='disable'):
    try:
        logger.info(f"Starting model training with GFP={gfp}, FHE={fhe_mode}")
        
        # Validate FHE mode
        valid_fhe_modes = ['disable', 'simulate', 'execute']
        if fhe_mode not in valid_fhe_modes:
            raise ValueError(f"Invalid FHE mode. Must be one of {valid_fhe_modes}")
        
        # Ensure directories exist
        ensure_directories()
        
        # Check if data file exists and is accessible
        check_data_file()
        
        # Load and prepare data
        logger.info(f"Loading data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        
        # Verify required columns exist
        missing_columns = [col for col in FEATURES + [TARGET] if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in data file: {missing_columns}")
        
        X = df[FEATURES]
        y = df[TARGET]
        
        # Add EdgeID for GFP if enabled
        if gfp:
            logger.info("Adding EdgeID for GFP enrichment")
            X = add_edge_id(X)
            logger.info("Applying GFP enrichment")
            X = pd.concat([X, gfp_enrichment(X)], axis=1)
            # Remove EdgeID after enrichment
            X = X.drop('EdgeID', axis=1)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        logger.info("Training XGBoost model")
        clf = XGBClassifier(n_jobs=1, n_bits=3)
        clf.fit(X_train, y_train)
        
        # Compile for FHE if needed
        if fhe_mode != 'disable':
            logger.info(f"Compiling model for FHE mode: {fhe_mode}")
            clf.compile(X_train)
        
        # Save model and scaler only if FHE is disabled
        if fhe_mode == 'disable':
            logger.info(f"Saving model to {MODEL_PATH}")
            joblib.dump(clf, MODEL_PATH)
            logger.info(f"Saving scaler to {SCALER_PATH}")
            joblib.dump(scaler, SCALER_PATH)
        else:
            logger.info("Skipping model save: FHE mode enabled (model not serializable)")
        
        logger.info("Model training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in train_and_save_model: {str(e)}")
        raise

def predict_from_input(input_dict, gfp=False, fhe_mode='disable'):
    try:
        logger.info(f"Making prediction with GFP={gfp}, FHE={fhe_mode}")
        
        # Validate FHE mode
        valid_fhe_modes = ['disable', 'simulate', 'execute']
        if fhe_mode not in valid_fhe_modes:
            raise ValueError(f"Invalid FHE mode. Must be one of {valid_fhe_modes}")
        
        # Validate input data
        validate_input_data(input_dict)
        
        # Ensure directories exist
        ensure_directories()
        
        # Check if model and scaler exist
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Please train the model first."
            )
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"Scaler file not found at {SCALER_PATH}. "
                "Please train the model first."
            )
        
        # Load model and scaler
        logger.info(f"Loading model from {MODEL_PATH}")
        clf = joblib.load(MODEL_PATH)
        logger.info(f"Loading scaler from {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
        
        # Prepare input data
        X = pd.DataFrame([input_dict])[FEATURES]
        
        # Add EdgeID and apply GFP if enabled
        if gfp:
            logger.info("Adding EdgeID for GFP enrichment")
            X = add_edge_id(X)
            logger.info("Applying GFP enrichment")
            X = pd.concat([X, gfp_enrichment(X)], axis=1)
            # Remove EdgeID after enrichment
            X = X.drop('EdgeID', axis=1)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        if fhe_mode == 'disable':
            pred = clf.predict(X_scaled, fhe='disable')[0]
        else:
            pred = clf.predict(X_scaled, fhe=fhe_mode)[0]
        
        logger.info(f"Prediction completed: {pred}")
        return int(pred)
        
    except Exception as e:
        logger.error(f"Error in predict_from_input: {str(e)}")
        raise 