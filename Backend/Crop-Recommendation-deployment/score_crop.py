import warnings
warnings.filterwarnings('ignore')

import json
import logging
import pandas as pd
import numpy as np
import joblib
import shap

logger = logging.getLogger(__name__)

class CropRecommendationModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.explainer = None
        self.feature_meanings = None
        
    def init(self):
        try:
            # Load model and preprocessing objects
            self.model = joblib.load('best_model_XGBoost.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.label_encoder = joblib.load('label_encoder.pkl')
            self.feature_columns = joblib.load('feature_names.pkl')
            
            # Initialize SHAP explainer with the original background data
            self._init_shap_explainer()
            
            # Feature meanings for explanations
            self.feature_meanings = {
                'N': 'Nitrogen level in soil',
                'P': 'Phosphorus level in soil',
                'K': 'Potassium level in soil',
                'humidity': 'Humidity level in the air',
                'rainfall': 'Amount of rainfall',
                'ph_rain': 'Interaction between rainfall and soil acidity (pH × rainfall)',
                'temp_rain': 'Interaction between temperature and rainfall',
                'NPK_Avg_Soil_Fertility': 'Average soil fertility (based on N, P, and K)',
                'NP_Ratio': 'Nitrogen-to-Phosphorus ratio (N ÷ P)',
                'THI': 'Combined effect of temperature and humidity on the environment'
            }
            
            logger.info("✅ Crop model initialized successfully with original SHAP explainer")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in crop model initialization: {str(e)}")
            return False
    
    def _init_shap_explainer(self):
        """Initialize SHAP explainer with the original background data"""
        try:
            # Load the original X_background from the notebook
            X_background = joblib.load('X_background.pkl')
            
            # Use the exact same SHAP explainer as in the notebook
            self.explainer = shap.TreeExplainer(
                self.model, 
                data=X_background,
                feature_perturbation="interventional"
            )
            logger.info("✅ SHAP explainer initialized with original background data")
            
        except Exception as e:
            logger.error(f"❌ Failed to load X_background.pkl: {str(e)}")
            # Fallback to alternative method if X_background is not available
            self._init_shap_explainer_fallback()
    
    def _init_shap_explainer_fallback(self):
        """Fallback method if X_background.pkl is not available"""
        try:
            logger.warning("⚠️ Using fallback SHAP explainer with generated background data")
            
            # Create realistic background data based on typical crop data ranges
            background_data = []
            
            for _ in range(100):
                sample = {
                    'N': np.random.uniform(0, 100),
                    'P': np.random.uniform(0, 100),
                    'K': np.random.uniform(0, 100),
                    'temperature': np.random.uniform(10, 40),
                    'humidity': np.random.uniform(20, 90),
                    'ph': np.random.uniform(5.0, 8.0),
                    'rainfall': np.random.uniform(50, 300)
                }
                
                # Apply the same feature engineering
                processed = self._engineer_features(sample)
                background_data.append(list(processed.values()))
            
            background_df = pd.DataFrame(background_data, columns=self.feature_columns)
            
            self.explainer = shap.TreeExplainer(
                self.model, 
                data=background_df,
                feature_perturbation="interventional"
            )
            logger.info("✅ Fallback SHAP explainer initialized")
            
        except Exception as e:
            logger.error(f"❌ Fallback SHAP explainer also failed: {str(e)}")
            self.explainer = None
    
    def _engineer_features(self, input_data):
        """Feature engineering - same as in training"""
        return {
            'temp_rain': input_data['temperature'] * input_data['rainfall'],
            'ph_rain': input_data['ph'] * input_data['rainfall'],
            'K': input_data['K'],
            'rainfall': input_data['rainfall'],
            'N': input_data['N'],
            'P': input_data['P'],
            'NPK_Avg_Soil_Fertility': (input_data['N'] + input_data['P'] + input_data['K']) / 3,
            'humidity': input_data['humidity'],
            'NP_Ratio': input_data['N'] / input_data['P'] if input_data['P'] != 0 else 0,
            'THI': (input_data['temperature'] * input_data['humidity']) / 100
        }
    
    def preprocess_input(self, input_data):
        """Preprocess input data to match training format"""
        try:
            # Create engineered features using the same method as training
            processed_data = self._engineer_features({
                'N': input_data['nitrogen'],
                'P': input_data['phosphorus'],
                'K': input_data['potassium'],
                'temperature': input_data['temperature'],
                'humidity': input_data['humidity'],
                'ph': input_data['ph'],
                'rainfall': input_data['rainfall']
            })
            
            # Create DataFrame with correct column order
            input_df = pd.DataFrame([processed_data], columns=self.feature_columns)
            
            # Scale the features for prediction
            scaled_features = self.scaler.transform(input_df)
            
            return scaled_features, processed_data, input_df
            
        except Exception as e:
            logger.error(f"Error in input preprocessing: {str(e)}")
            raise
    
    def _extract_single_shap_vector(self, shap_raw, top_class_index, num_features, num_classes=None):
        """Extract SHAP values for the top predicted class - same as notebook"""
        try:
            arr = np.asarray(shap_raw)

            if arr.ndim == 1 and arr.shape[0] == num_features:
                return arr

            if arr.ndim == 2 and arr.shape[1] == num_features:
                return arr[0]

            shape = arr.shape
            axes_with_features = [i for i, s in enumerate(shape) if s == num_features]
            if not axes_with_features:
                raise ValueError(f"Can't find an axis matching num_features={num_features} in shap array with shape {shape}")

            feat_axis = axes_with_features[0]

            idx = []
            for i, s in enumerate(shape):
                if i == feat_axis:
                    idx.append(slice(None))
                else:
                    if (num_classes is not None and s == num_classes):
                        idx.append(top_class_index)
                    else:
                        idx.append(0)

            vec = arr[tuple(idx)].reshape(-1)
            if vec.shape[0] != num_features:
                raise ValueError(f"Extracted SHAP vector length {vec.shape[0]} doesn't match num_features={num_features}")
            return vec
            
        except Exception as e:
            logger.warning(f"SHAP extraction warning: {e}")
            return np.zeros(num_features)
    
    def run(self, input_data):
        try:
            # Preprocess input
            scaled_data, processed_features, input_df = self.preprocess_input(input_data)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(scaled_data)[0]
            
            # Get ONLY the top recommendation
            top_index = np.argmax(prediction_proba)
            crop = self.label_encoder.inverse_transform([top_index])[0]
            confidence = float(prediction_proba[top_index])
            
            # SHAP Explanation - using the same method as notebook
            explanation_data = self._get_shap_explanation(input_df, top_index, processed_features, crop)
            
            # Result format
            result = {
                "crop": crop,
                "confidence": confidence,
                "suitability": f"{confidence:.1%}",
                "key_factors": explanation_data["explanations"] if "explanations" in explanation_data else [],
                "input_summary": {
                    "nitrogen": input_data['nitrogen'],
                    "phosphorus": input_data['phosphorus'], 
                    "potassium": input_data['potassium'],
                    "temperature": input_data['temperature'],
                    "humidity": input_data['humidity'],
                    "ph": input_data['ph'],
                    "rainfall": input_data['rainfall']
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during crop prediction: {str(e)}")
            raise
    
    def _get_shap_explanation(self, input_df, top_index, processed_features, crop):
        """Generate SHAP explanation using the same method as notebook"""
        try:
            if self.explainer is None:
                return {"explanations": ["Detailed explanation not available"]}
            
            # Calculate SHAP values using the same method as notebook
            shap_values_raw = self.explainer.shap_values(input_df)
            
            # Extract SHAP values for top class using the same function as notebook
            shap_values_for_sample = self._extract_single_shap_vector(
                shap_values_raw,
                top_class_index=top_index,
                num_features=len(self.feature_columns),
                num_classes=len(self.model.classes_)
            )
            
            # Get top 3 influential features
            feature_importance = []
            for i, feature in enumerate(self.feature_columns):
                importance = {
                    'feature': feature,
                    'shap_value': float(shap_values_for_sample[i]),
                    'value': float(processed_features[feature]),
                    'meaning': self.feature_meanings.get(feature, feature)
                }
                feature_importance.append(importance)
            
            # Sort by absolute SHAP value
            feature_importance.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            top_features = feature_importance[:3]
            
            # Generate explanation text - same format as notebook
            explanations = []
            for feature in top_features:
                direction = "increased" if feature['shap_value'] > 0 else "decreased"
                explanation = f"{feature['meaning']} (value: {feature['value']}) {direction} the likelihood of recommending {crop}."
                explanations.append(explanation)
            
            return {
                "explanations": explanations
            }
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {"explanations": ["Explanation generation failed"]}

# Initialize model instance
crop_model = CropRecommendationModel()

def init():
    return crop_model.init()

def run(input_data):
    try:
        result = crop_model.run(input_data)
        return result
    except Exception as e:
        return {"error": str(e), "status": "error"}
