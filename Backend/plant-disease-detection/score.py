import json
import logging
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import io

logger = logging.getLogger(__name__)

class PlantDiseaseModel:
    def __init__(self):
        self.model = None
        self.device = None
        self.categories = None
        self.transform = None
        self.disease_classes = None
        self.healthy_classes = None

    def init(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load categories
            with open('categories.json', 'r') as f:
                self.categories = json.load(f)
            
            # Define transforms
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Load model
            self.model = self._load_model()
            
            # Define disease/healthy classes
            self.disease_classes = [i for i, c in enumerate(self.categories) if "healthy" not in c.lower()]
            self.healthy_classes = [i for i, c in enumerate(self.categories) if "healthy" in c.lower()]
            
            return True
            
        except Exception as e:
            logger.error(f"Error in model initialization: {str(e)}")
            return False

    def _load_model(self):
        try:
            from torchvision import models
            import torch.nn as nn
            
            model_path = "best_resnet50_model_by_f1.pth"
            
            # Load model architecture
            model = models.resnet50(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.6),
                nn.Linear(num_features, len(self.categories))
            )
            
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def preprocess_image(self, image_data):
        try:
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = Image.open(image_data)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_tensor = self.transform(image).unsqueeze(0)
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            raise

    def run(self, image_data):
        try:
            input_tensor = self.preprocess_image(image_data)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get top probability from each group
                max_prob_disease = probabilities[0][self.disease_classes].max().item()
                max_prob_healthy = probabilities[0][self.healthy_classes].max().item()

                # Compare and determine status
                if max_prob_disease > max_prob_healthy:
                    status = "Diseased"
                    overall_confidence = max_prob_disease
                else:
                    status = "Healthy"
                    overall_confidence = max_prob_healthy

                result = {
                    "status": status,
                    "overall_confidence": float(overall_confidence)
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise


# Global model instance
model = PlantDiseaseModel()

def init():
    return model.init()

def run(raw_data):
    try:
        result = model.run(raw_data)
        return result
    except Exception as e:
        return {"error": str(e)}
