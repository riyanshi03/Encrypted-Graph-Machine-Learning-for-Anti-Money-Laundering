# Encrypted Graph Machine Learning for Anti-Money Laundering

This is my Final year project. It implements a Privacy-Preserving Machine Learning (PPML) solution for Anti-Money Laundering (AML) detection using XGBoost with Fully Homomorphic Encryption (FHE) and Graph Feature Processing (GFP).

## Architecture

The system consists of three main components:

1. **Frontend**: A Streamlit web application for user interaction
2. **Backend**: A Flask API server handling model training and prediction
3. **Concrete ML**: A service for FHE operations

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.10 or higher
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd aml_pipeline
```

2. Build and start the containers:
```bash
docker-compose up --build
```

The services will be available at:
- Frontend: http://localhost:8502
- Backend: http://localhost:8000

## Usage

### Training the Model

1. Open your browser and navigate to `http://localhost:8502`
2. Click on "Train Model" and select your desired options:
   - Enable/disable GFP enrichment
   - Choose FHE mode (disable/simulate/execute)

### Making Predictions

1. Fill in the transaction details in the form:
   - Source Account ID
   - Target Account ID
   - Timestamp
   - Amount Sent
   - Sent Currency
   - Amount Received
   - Receiving Currency
   - Payment Format
2. Select GFP and FHE options
3. Click "Predict" to get the result

## Development

### Running Locally

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

3. Start the backend server:
```bash
cd backend
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

4. Start the frontend:
```bash
cd frontend
streamlit run app.py --server.port 8502
```

## API Endpoints

### Backend API (http://localhost:8000)

- `POST /train`: Train the model with specified options
- `POST /predict`: Make predictions on new transactions
- `GET /health`: Health check endpoint

## License

[Your License Here]
