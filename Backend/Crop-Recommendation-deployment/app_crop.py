from flask import Flask, request, jsonify
import logging
import sys
import os
from flask_cors import CORS

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from score_crop import CropRecommendationModel

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize crop model
crop_model = CropRecommendationModel()
init_success = crop_model.init()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    return jsonify({
        "message": "Crop Recommendation API",
        "status": "running",
        "model_loaded": init_success
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_initialized": init_success
    })

@app.route('/recommend', methods=['POST'])
def recommend_crop():
    try:
        if not init_success:
            return jsonify({"error": "Crop model not initialized"}), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Required parameters 
        required_params = [
            'nitrogen', 'phosphorus', 'potassium', 
            'temperature', 'humidity', 'ph', 'rainfall'
        ]
        
        # Check if all required parameters are present
        for param in required_params:
            if param not in data:
                return jsonify({"error": f"Missing parameter: {param}"}), 400
        
        # Prepare input data
        input_data = {
            'nitrogen': float(data['nitrogen']),
            'phosphorus': float(data['phosphorus']),
            'potassium': float(data['potassium']),
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'ph': float(data['ph']),
            'rainfall': float(data['rainfall'])
        }
        
        logger.info(f"Received input: {input_data}")
        
        # Get prediction
        result = crop_model.run(input_data)

        logger.info(f"Prediction: {result['crop']} ({result['confidence']:.2%})")
        
        return jsonify(result)
        
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        return jsonify({"error": f"Invalid parameter type: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/crops', methods=['GET'])
def get_available_crops():
    """Get list of all available crops in the model"""
    try:
        if not init_success:
            return jsonify({"error": "Model not initialized"}), 500
            
        crops = crop_model.label_encoder.classes_.tolist()
        return jsonify({"available_crops": crops})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860)) 
    app.run(host='0.0.0.0', port=port, debug=False)
