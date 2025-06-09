import os

# Frontend configuration
FRONTEND_PORT = int(os.environ.get('FRONTEND_PORT', 8502))
FRONTEND_HOST = os.environ.get('FRONTEND_HOST', '0.0.0.0')

# Backend configuration
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://backend:5000')

# ConcreteML configuration
CONCRETE_ML_URL = os.environ.get('CONCRETE_ML_URL', 'http://concrete_ml:8000')

# API endpoints
API_ENDPOINTS = {
    'health': f'{BACKEND_URL}/health',
    'train': f'{BACKEND_URL}/train',
    'predict': f'{BACKEND_URL}/predict',
    'concrete_ml_health': f'{CONCRETE_ML_URL}/health'
} 