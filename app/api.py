from flask import Blueprint, jsonify, request
import sys, os

# Correctly set the path to the project root to import the pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.prediction_pipeline import HeartDiseasePredictionPipeline

# Create a Blueprint for the API
api_bp = Blueprint('api', __name__)

# Initialize the pipeline once
pipeline = HeartDiseasePredictionPipeline()

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify that the application is running.
    """
    return jsonify({"status": "ok"}), 200

@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint that accepts patient data as JSON and returns a prediction.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    patient_data = request.get_json()
    result = pipeline.predict(patient_data)

    if not result.get('success'):
        errors = result.get('errors') or [result.get('error', 'Unknown error')]
        return jsonify({"success": False, "errors": errors}), 400

    return jsonify({"success": True, "result": result}), 200

