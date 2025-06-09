import streamlit as st
import requests
import os
from urllib.parse import urljoin

# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://backend:5000')  # Use service name in docker-compose

# Custom CSS for professional styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Title styling */
    h1 {
        color: #1E3A8A;
        font-weight: 700;
        font-size: 2.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
    }
    
    /* Header styling */
    h2 {
        color: #2563EB;
        font-weight: 600;
        font-size: 1.8rem;
        margin-top: 2rem;
    }
    
    /* Form styling */
    .stForm {
        background-color: #F8FAFC;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        border: 1px solid #E5E7EB;
        border-radius: 4px;
        padding: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2563EB;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 1rem;
        border-radius: 4px;
        border-left: 4px solid #059669;
    }
    
    /* Error message styling */
    .stError {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 1rem;
        border-radius: 4px;
        border-left: 4px solid #DC2626;
    }
    
    /* Checkbox styling */
    .stCheckbox {
        color: #4B5563;
    }
    
    /* Selectbox styling */
    .stSelectbox {
        color: #4B5563;
    }
    
    /* Info box styling */
    .stInfo {
        background-color: #EFF6FF;
        color: #1E40AF;
        padding: 1rem;
        border-radius: 4px;
        border-left: 4px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

def make_api_request(endpoint, method='POST', json_data=None):
    """Make API request with error handling"""
    try:
        url = urljoin(BACKEND_URL, endpoint)
        headers = {'Content-Type': 'application/json'}
        response = requests.request(method, url, json=json_data, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

st.title("PPML AML Detection (XGBoost + FHE + GFP)")

# Model Training Section
st.header("Train Model")
with st.form("training_form"):
    gfp = st.checkbox("Enable GFP Enrichment")
    fhe_mode = st.selectbox("FHE Mode", ["disable", "simulate", "execute"])
    train_submitted = st.form_submit_button("Train Model")

if train_submitted:
    with st.spinner("Training model..."):
        options = {
            "gfp": bool(gfp),
            "fhe_mode": str(fhe_mode)
        }
        response = make_api_request('/train', method='POST', json_data=options)
        if response and response.get('status') == 'success':
            st.session_state.model_trained = True
            st.success("Model trained successfully!")
        else:
            st.error("Training failed. Please check the logs for details.")

# Prediction Section
st.header("Predict Transaction")
with st.form("prediction_form"):
    source = st.number_input("Source Account ID", min_value=0)
    target = st.number_input("Target Account ID", min_value=0)
    timestamp = st.number_input("Timestamp", min_value=0)
    amount_sent = st.number_input("Amount Sent", min_value=0.0, format="%f")
    sent_currency = st.number_input("Sent Currency (as int)", min_value=0)
    amount_received = st.number_input("Amount Received", min_value=0.0, format="%f")
    receiving_currency = st.number_input("Receiving Currency (as int)", min_value=0)
    payment_format = st.number_input("Payment Format (as int)", min_value=0)
    predict_submitted = st.form_submit_button("Predict")

if predict_submitted:
    if not st.session_state.model_trained:
        st.warning("Please train the model first before making predictions.")
    else:
        with st.spinner("Making prediction..."):
            data = {
                "features": {
                    "SourceAccountId": int(source),
                    "TargetAccountId": int(target),
                    "Timestamp": int(timestamp),
                    "Amount Sent": float(amount_sent),
                    "Sent Currency": int(sent_currency),
                    "Amount Received": float(amount_received),
                    "Receiving Currency": int(receiving_currency),
                    "Payment Format": int(payment_format)
                },
                "gfp": gfp,
                "fhe_mode": fhe_mode
            }
            response = make_api_request('/predict', json_data=data)
            if response and response.get('status') == 'success':
                prediction = response.get('prediction')
                label = response.get('label', 'Laundering' if prediction == 1 else 'Not Laundering')
                st.success(f"Prediction: {label}")
            else:
                st.error("Prediction failed. Please check the logs for details.") 