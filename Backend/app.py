from flask import Flask, request, jsonify
import io
import base64
from PIL import Image
import logging
import sys
import os

# Add current directory to path to import score
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your model scoring
from score import PlantDiseaseModel

# Initialize Flask app
app = Flask(__name__)

# Initialize model
model = PlantDiseaseModel()
init_success = model.init()

@app.route('/')
def home():
    return jsonify({
        "message": "Plant Disease Detection API",
        "status": "running",
        "model_loaded": init_success
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_initialized": init_success
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not init_success:
            return jsonify({"error": "Model not initialized"}), 500
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.content_type.startswith('image/'):
            return jsonify({"error": "File must be an image"}), 400
        
        image_data = file.read()
        result = model.run(image_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860)) 
    app.run(host='0.0.0.0', port=port, debug=False)
