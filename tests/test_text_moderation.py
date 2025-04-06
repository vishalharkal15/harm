import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_manager import ModelManager, TextModel
from src.core.content_processor import ContentProcessor


class MockTextModel(TextModel):
    """Mock text model for testing"""
    
    def __init__(self):
        super().__init__()
        self.loaded = True
        self.health_status = True
    
    def _load_model(self):
        """Mock implementation"""
        self.loaded = True
        self.categories = [
            "hate_speech",
            "harassment", 
            "violence",
            "self_harm",
            "sexual",
            "terrorism",
            "misinformation",
        ]
    
    def _predict(self, text, context=None):
        """Mock prediction that returns different results based on input text"""
        result = {category: 0.1 for category in self.categories}
        
        # Test cases for different harmful content types
        if "hate" in text.lower():
            result["hate_speech"] = 0.85
        
        if "attack" in text.lower() or "kill" in text.lower():
            result["violence"] = 0.92
            
        if "fake news" in text.lower():
            result["misinformation"] = 0.78
            
        if "nude" in text.lower() or "sex" in text.lower():
            result["sexual"] = 0.89
            
        return result


class TestTextModeration(unittest.TestCase):
    """Test the text moderation functionality"""
    
    def setUp(self):
        """Set up the test environment"""
        # Create a mock model manager that returns our mock text model
        self.model_manager = MagicMock()
        self.model_manager.get_model.return_value = MockTextModel()
        
        # Create the content processor with our mock model manager
        self.content_processor = ContentProcessor(self.model_manager)
    
    def test_safe_content(self):
        """Test that safe content is correctly identified"""
        text = "This is a normal message about programming and technology."
        result = self.content_processor.process_text(text)
        
        self.assertFalse(result["is_harmful"])
        self.assertLess(result["confidence"], 0.7)
        
    def test_hate_speech(self):
        """Test that hate speech is correctly identified"""
        text = "I hate all people from that country and wish they would disappear."
        result = self.content_processor.process_text(text)
        
        self.assertTrue(result["is_harmful"])
        self.assertEqual(result["categories"]["hate_speech"], 0.85)
        self.assertGreaterEqual(result["confidence"], 0.7)
        
    def test_violent_content(self):
        """Test that violent content is correctly identified"""
        text = "I'm going to attack and kill everyone at that location tomorrow."
        result = self.content_processor.process_text(text)
        
        self.assertTrue(result["is_harmful"])
        self.assertEqual(result["categories"]["violence"], 0.92)
        self.assertGreaterEqual(result["confidence"], 0.7)
        
    def test_misinformation(self):
        """Test that misinformation is correctly identified"""
        text = "This is fake news created to mislead people about important facts."
        result = self.content_processor.process_text(text)
        
        self.assertTrue(result["is_harmful"])
        self.assertEqual(result["categories"]["misinformation"], 0.78)
        self.assertGreaterEqual(result["confidence"], 0.7)
        
    def test_sexual_content(self):
        """Test that sexual content is correctly identified"""
        text = "Here are explicit nude photos and sex videos that I'm sharing."
        result = self.content_processor.process_text(text)
        
        self.assertTrue(result["is_harmful"])
        self.assertEqual(result["categories"]["sexual"], 0.89)
        self.assertGreaterEqual(result["confidence"], 0.7)
        
    def test_multiple_categories(self):
        """Test content that falls into multiple harmful categories"""
        text = "I hate those people and will attack them. This is fake news about nude photos."
        result = self.content_processor.process_text(text)
        
        self.assertTrue(result["is_harmful"])
        self.assertGreaterEqual(result["categories"]["hate_speech"], 0.7)
        self.assertGreaterEqual(result["categories"]["violence"], 0.7)
        self.assertGreaterEqual(result["categories"]["misinformation"], 0.7)
        self.assertGreaterEqual(result["confidence"], 0.7)


if __name__ == "__main__":
    unittest.main() 