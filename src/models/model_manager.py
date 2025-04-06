import os
import json
import time
from typing import Dict, Any, Optional, List, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image

from src.utils.logger import setup_logger
from src.config.model_config import MODEL_CONFIG

logger = setup_logger("model_manager")

class BaseModel:
    """Base class for all models"""
    
    def __init__(self):
        self.name = "base_model"
        self.type = "base"
        self.loaded = False
        self.health_status = False
        self.loading_time = 0
        self.last_prediction_time = 0
    
    def load(self):
        """Load the model into memory"""
        start_time = time.time()
        try:
            self._load_model()
            self.loaded = True
            self.health_status = True
            self.loading_time = time.time() - start_time
            logger.info(f"Model {self.name} loaded successfully in {self.loading_time:.2f}s")
        except Exception as e:
            self.loaded = False
            self.health_status = False
            logger.error(f"Failed to load model {self.name}: {str(e)}")
            raise
    
    def _load_model(self):
        """Implementation-specific model loading"""
        raise NotImplementedError("Subclasses must implement _load_model")
    
    def predict(self, *args, **kwargs):
        """Make a prediction"""
        if not self.loaded:
            self.load()
        
        start_time = time.time()
        result = self._predict(*args, **kwargs)
        self.last_prediction_time = time.time() - start_time
        
        return result
    
    def _predict(self, *args, **kwargs):
        """Implementation-specific prediction logic"""
        raise NotImplementedError("Subclasses must implement _predict")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the model"""
        return {
            "name": self.name,
            "type": self.type,
            "loaded": self.loaded,
            "healthy": self.health_status,
            "loading_time": self.loading_time,
            "last_prediction_time": self.last_prediction_time
        }


class TextModel(BaseModel):
    """Model for text content moderation"""
    
    def __init__(self, model_path: str = None):
        super().__init__()
        self.name = "text_model"
        self.type = "text"
        self.model_path = model_path or MODEL_CONFIG.get("text", {}).get("path", "bert-base-uncased")
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.categories = MODEL_CONFIG.get("text", {}).get("categories", [])
    
    def _load_model(self):
        """Load the text classification model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load text model: {str(e)}")
            raise
    
    def _predict(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Predict harmful content categories in text"""
        # For a real implementation, this would use the actual model
        # Here we simulate the prediction for demonstration purposes
        
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Map to categories
        result = {}
        for i, category in enumerate(self.categories):
            if i < len(probabilities):
                result[category] = float(probabilities[i])
        
        return result


class ImageModel(BaseModel):
    """Model for image content moderation"""
    
    def __init__(self, model_path: str = None):
        super().__init__()
        self.name = "image_model"
        self.type = "image"
        self.model_path = model_path or MODEL_CONFIG.get("image", {}).get("path", "efficientnet-b0")
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.categories = MODEL_CONFIG.get("image", {}).get("categories", [])
    
    def _load_model(self):
        """Load the image classification model"""
        try:
            # In a real implementation, load the actual model
            # Here we just simulate loading
            self.model = "image_model_placeholder"
        except Exception as e:
            logger.error(f"Failed to load image model: {str(e)}")
            raise
    
    def _predict(self, image: Image.Image, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Predict harmful content categories in image"""
        # For a real implementation, this would use the actual model
        # Here we simulate the prediction for demonstration purposes
        
        result = {}
        for category in self.categories:
            # Simulate a prediction by generating a random score
            # In a real implementation, we would use the model to compute these scores
            result[category] = 0.1  # Default low probability for demonstration
        
        # Add some example detections (this would be from the actual model in production)
        if "sexual" in result:
            result["sexual"] = 0.2
        if "violence" in result:
            result["violence"] = 0.15
            
        return result


class VideoModel(BaseModel):
    """Model for video content moderation"""
    
    def __init__(self, model_path: str = None):
        super().__init__()
        self.name = "video_model"
        self.type = "video"
        self.model_path = model_path or MODEL_CONFIG.get("video", {}).get("path", "video_model")
        self.frame_model = None
        self.temporal_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.categories = MODEL_CONFIG.get("video", {}).get("categories", [])
    
    def _load_model(self):
        """Load the video analysis model"""
        try:
            # In a real implementation, load the actual models
            # Here we just simulate loading
            self.frame_model = ImageModel()
            self.frame_model.load()
            self.temporal_model = "temporal_model_placeholder"
        except Exception as e:
            logger.error(f"Failed to load video model: {str(e)}")
            raise
    
    def predict_frame(self, frame: Image.Image, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Predict harmful content in a single video frame"""
        return self.frame_model.predict(frame, context)
    
    def _predict(self, video_path: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Predict harmful content categories in video
        
        NOTE: Full video prediction would be implemented here.
        For API usage, the ContentProcessor handles frame extraction and calls predict_frame.
        """
        # For a real implementation, this would analyze the full video
        # Here we just return a placeholder
        return {category: 0.1 for category in self.categories}


class MultimodalModel(BaseModel):
    """Model for fusing and analyzing multimodal content"""
    
    def __init__(self):
        super().__init__()
        self.name = "multimodal_model"
        self.type = "multimodal"
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self):
        """Load the multimodal fusion model"""
        try:
            # In a real implementation, load any necessary fusion models
            # Here we just simulate loading
            self.loaded = True
        except Exception as e:
            logger.error(f"Failed to load multimodal model: {str(e)}")
            raise
    
    def _predict(self, *args, **kwargs):
        """Not used directly - instead use fuse_results"""
        raise NotImplementedError("Use fuse_results instead")
    
    def fuse_results(self, results: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fuse results from multiple modalities"""
        # In a real implementation, this would use a sophisticated fusion strategy
        # Here we implement a simple one that prioritizes the most confident harmful prediction
        
        # Track if any modality found harmful content
        any_harmful = False
        max_confidence = 0.0
        max_category = ""
        max_modality = ""
        all_categories = {}
        
        for modality, result in results.items():
            if result.get("is_harmful", False):
                any_harmful = True
                
                # Find the most confident harmful prediction across all modalities
                if result.get("confidence", 0) > max_confidence:
                    max_confidence = result.get("confidence", 0)
                    max_category = result.get("categories", {})
                    max_modality = modality
            
            # Collect all category scores from all modalities
            for category, score in result.get("categories", {}).items():
                if category not in all_categories:
                    all_categories[category] = []
                all_categories[category].append(score)
        
        # Aggregate category scores (max pooling across modalities)
        aggregated_categories = {}
        for category, scores in all_categories.items():
            aggregated_categories[category] = max(scores) if scores else 0.0
        
        # Create fusion result
        fusion_result = {
            "is_harmful": any_harmful,
            "categories": aggregated_categories,
            "confidence": max_confidence,
            "primary_modality": max_modality if any_harmful else None,
            "modalities_analyzed": list(results.keys())
        }
        
        # Generate explanation
        if any_harmful:
            # Get max category from aggregated scores
            max_cat = max(aggregated_categories.items(), key=lambda x: x[1])
            cat_name, score = max_cat
            
            explanation = f"Multimodal analysis detected harmful content ({cat_name}) with {score:.1%} confidence."
            explanation += f" Primary evidence found in {max_modality} content."
            
            if len(results) > 1:
                # Check for corroborating evidence
                corroborating = [mod for mod, res in results.items() 
                               if mod != max_modality and res.get("is_harmful", False)]
                if corroborating:
                    explanation += f" Corroborating evidence in {', '.join(corroborating)}."
        else:
            explanation = "No harmful content detected across any analyzed modalities."
        
        fusion_result["explanation"] = explanation
        return fusion_result


class ModelManager:
    """Manages loading, caching, and accessing AI models for content moderation"""
    
    def __init__(self):
        self.models = {}
        self.config = MODEL_CONFIG
        self.default_models = {
            "text": TextModel,
            "image": ImageModel,
            "video": VideoModel,
            "multimodal": MultimodalModel
        }
        
        logger.info("ModelManager initialized")
    
    def get_model(self, model_type: str) -> BaseModel:
        """Get a model of the specified type, loading it if necessary"""
        if model_type not in self.models:
            logger.info(f"Loading model of type: {model_type}")
            
            # Create the model instance
            model_class = self.default_models.get(model_type)
            if not model_class:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model = model_class()
            model.load()
            self.models[model_type] = model
            
        return self.models[model_type]
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health status of all loaded models"""
        results = {
            "all_healthy": True,
            "models": {}
        }
        
        for model_type, model in self.models.items():
            status = model.get_status()
            results["models"][model_type] = status
            if not status["healthy"]:
                results["all_healthy"] = False
        
        return results
    
    def reload_model(self, model_type: str) -> bool:
        """Force a reload of a specific model"""
        if model_type in self.models:
            try:
                logger.info(f"Reloading model: {model_type}")
                model = self.models[model_type]
                model.load()
                return model.health_status
            except Exception as e:
                logger.error(f"Failed to reload model {model_type}: {str(e)}")
                return False
        else:
            try:
                self.get_model(model_type)
                return True
            except Exception:
                return False 