from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import os
from ml_pipeline import train_and_save_model, predict_from_input
import traceback
from datetime import datetime
import sys

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log')

# Configure logging with fallback to stdout if file logging fails
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
except Exception as e:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.warning(f"Could not set up file logging: {str(e)}. Falling back to stdout logging.")

logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configure CORS to allow all origins in development
CORS(app, resources={r"/*": {"origins": "*"}})

# Enable debug mode if FLASK_DEBUG is set
app.debug = os.environ.get('FLASK_DEBUG', '0') == '1'

# Get port and host from environment variables
PORT = int(os.environ.get('PORT', 5000))
HOST = os.environ.get('HOST', '0.0.0.0')
CONCRETE_ML_URL = os.environ.get('CONCRETE_ML_URL', 'http://concrete_ml:8000')

def validate_prediction_input(data):
    """Validate prediction input data"""
    if not isinstance(data, dict):
        return False, "Input must be a JSON object"
    
    required_fields = ['features']
    if not all(field in data for field in required_fields):
        return False, "Missing required fields"
    
    features = data['features']
    if not isinstance(features, dict):
        return False, "Features must be a JSON object"
    
    required_features = [
        'SourceAccountId', 'TargetAccountId', 'Timestamp',
        'Amount Sent', 'Sent Currency', 'Amount Received',
        'Receiving Currency', 'Payment Format'
    ]
    if not all(field in features for field in required_features):
        return False, "Missing required features"
    
    # Validate data types
    try:
        if not isinstance(features['Amount Sent'], (int, float)) or features['Amount Sent'] < 0:
            return False, "Amount Sent must be a non-negative number"
        if not isinstance(features['Amount Received'], (int, float)) or features['Amount Received'] < 0:
            return False, "Amount Received must be a non-negative number"
        if not isinstance(features['SourceAccountId'], (int, str)):
            return False, "SourceAccountId must be an integer or string"
        if not isinstance(features['TargetAccountId'], (int, str)):
            return False, "TargetAccountId must be an integer or string"
        if not isinstance(features['Sent Currency'], str):
            return False, "Sent Currency must be a string"
        if not isinstance(features['Receiving Currency'], str):
            return False, "Receiving Currency must be a string"
        if not isinstance(features['Payment Format'], str):
            return False, "Payment Format must be a string"
        # Validate timestamp
        datetime.strptime(features['Timestamp'], '%Y-%m-%d %H:%M:%S')
    except ValueError as e:
        return False, f"Invalid data format: {str(e)}"
    
    return True, None

def validate_training_input(data):
    """Validate training input data"""
    if not isinstance(data, dict):
        return False, "Input must be a JSON object"
    
    # Validate FHE mode
    valid_fhe_modes = ['disable', 'simulate', 'execute']
    fhe_mode = data.get('fhe_mode', 'disable')
    if fhe_mode not in valid_fhe_modes:
        return False, f"Invalid FHE mode. Must be one of {valid_fhe_modes}"
    
    # Validate GFP flag
    if 'gfp' in data and not isinstance(data['gfp'], bool):
        return False, "GFP flag must be a boolean"
    
    return True, None

@app.route('/')
def index():
    """Root endpoint that returns API information"""
    return jsonify({
        "status": "running",
        "service": "backend",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "train": "/train",
            "predict": "/predict"
        }
    })

@app.route('/train', methods=['POST'])
def train():
    try:
        options = request.json or {}
        logger.info(f"Training request received with options: {options}")
        
        # Validate input
        is_valid, error_message = validate_training_input(options)
        if not is_valid:
            return jsonify({'status': 'error', 'message': error_message}), 400
        
        train_and_save_model(
            gfp=options.get('gfp', False),
            fhe_mode=options.get('fhe_mode', 'disable')
        )
        return jsonify({'status': 'success', 'message': 'Model trained and saved successfully'})
    except Exception as e:
        logger.error(f"Training error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        logger.info(f"Prediction request received: {input_data}")
        
        # Validate input
        is_valid, error_message = validate_prediction_input(input_data)
        if not is_valid:
            return jsonify({'status': 'error', 'message': error_message}), 400
        
        prediction = predict_from_input(
            input_data['features'],
            gfp=input_data.get('gfp', False),
            fhe_mode=input_data.get('fhe_mode', 'disable')
        )
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'label': 'Laundering' if prediction == 1 else 'Not Laundering'
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET', 'POST'])
def health_check():
    try:
        # Check if required directories exist
        required_dirs = ['logs', 'models', 'data']
        dir_status = {}
        for dir_name in required_dirs:
            dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dir_name)
            exists = os.path.exists(dir_path)
            dir_status[dir_name] = {
                'exists': exists,
                'path': dir_path
            }
            if not exists:
                os.makedirs(dir_path, exist_ok=True)
        
        # Check if model files exist
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model.joblib')
        scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'scaler.joblib')
        
        # Check if we can write to the log directory
        try:
            test_log_file = os.path.join(log_dir, 'test.log')
            with open(test_log_file, 'w') as f:
                f.write('test')
            os.remove(test_log_file)
            log_writeable = True
        except Exception as e:
            logger.error(f"Log directory not writeable: {str(e)}")
            log_writeable = False
        
        return jsonify({
            "status": "healthy",
            "service": "backend",
            "version": "1.0.0",
            "directories": dir_status,
            "model_files": {
                "model": {
                    "exists": os.path.exists(model_path),
                    "path": model_path
                },
                "scaler": {
                    "exists": os.path.exists(scaler_path),
                    "path": scaler_path
                }
            },
            "permissions": {
                "log_writeable": log_writeable
            }
        }), 200
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'The requested URL was not found on the server.'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'status': 'error',
        'message': 'The method is not allowed for the requested URL.'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error occurred.'
    }), 500

if __name__ == '__main__':
    logger.info(f"Starting backend server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=app.debug) 
    app.run(host='0.0.0.0', port=5000)
    