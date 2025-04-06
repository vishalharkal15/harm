import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

from src.utils.logger import setup_logger
from src.models.model_manager import ModelManager
from src.utils.helpers import format_results

logger = setup_logger("content_processor")

class ContentProcessor:
    """
    Main processor for handling different content types (text, image, video)
    and routing them to the appropriate models for analysis.
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
        # Default moderation categories
        self.categories = {
            "hate_speech": "Content expressing hatred or encouraging violence against a person or group",
            "harassment": "Content intending to intimidate, shame, or cause emotional distress",
            "violence": "Content depicting or threatening physical harm",
            "self_harm": "Content encouraging or depicting self-harm behaviors",
            "sexual": "Adult content including nudity or sexual acts",
            "child_exploitation": "Content exploiting or endangering minors",
            "terrorism": "Content promoting extremist acts or ideologies",
            "misinformation": "False or misleading information presented as fact",
            "spam": "Unwanted, repetitive content",
            "illegal_activity": "Content promoting or facilitating illegal activities"
        }
        
        logger.info("ContentProcessor initialized with all moderation categories")
    
    def process_text(self, text: str, context: Optional[Dict[str, Any]] = None, 
                    settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text content for harmful content
        
        Args:
            text: The text to analyze
            context: Optional contextual information
            settings: Optional processing settings
            
        Returns:
            Dictionary with moderation results
        """
        start_time = time.time()
        logger.info(f"Processing text content: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        try:
            # Get text analysis model
            model = self.model_manager.get_model("text")
            
            # Process settings
            threshold = self._get_threshold(settings)
            
            # Process the text
            categories_result = model.predict(text, context)
            
            # Determine if content is harmful based on thresholds
            is_harmful, max_category, confidence = self._evaluate_harmfulness(categories_result, threshold)
            
            # Generate explanation
            explanation = self._generate_explanation(is_harmful, max_category, confidence, categories_result)
            
            processing_time = time.time() - start_time
            
            result = {
                "is_harmful": is_harmful,
                "categories": categories_result,
                "explanation": explanation,
                "confidence": confidence,
                "processing_time": processing_time
            }
            
            logger.info(f"Text processing completed in {processing_time:.4f}s. Harmful: {is_harmful}")
            return result
            
        except Exception as e:
            logger.error(f"Error in text processing: {str(e)}")
            raise
    
    def process_image(self, image: bytes, context: Optional[Dict[str, Any]] = None, 
                     settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process image content for harmful content
        
        Args:
            image: The image bytes to analyze
            context: Optional contextual information
            settings: Optional processing settings
            
        Returns:
            Dictionary with moderation results
        """
        start_time = time.time()
        logger.info("Processing image content")
        
        try:
            # Convert bytes to image
            img = Image.open(BytesIO(image))
            
            # Get image analysis model
            model = self.model_manager.get_model("image")
            
            # Process settings
            threshold = self._get_threshold(settings)
            
            # Process the image
            categories_result = model.predict(img, context)
            
            # Determine if content is harmful based on thresholds
            is_harmful, max_category, confidence = self._evaluate_harmfulness(categories_result, threshold)
            
            # Generate explanation
            explanation = self._generate_explanation(is_harmful, max_category, confidence, categories_result)
            
            processing_time = time.time() - start_time
            
            result = {
                "is_harmful": is_harmful,
                "categories": categories_result,
                "explanation": explanation,
                "confidence": confidence,
                "processing_time": processing_time
            }
            
            logger.info(f"Image processing completed in {processing_time:.4f}s. Harmful: {is_harmful}")
            return result
            
        except Exception as e:
            logger.error(f"Error in image processing: {str(e)}")
            raise
    
    def process_video(self, video: bytes, context: Optional[Dict[str, Any]] = None, 
                     settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process video content for harmful content
        
        Args:
            video: The video bytes to analyze
            context: Optional contextual information
            settings: Optional processing settings
            
        Returns:
            Dictionary with moderation results
        """
        start_time = time.time()
        logger.info("Processing video content")
        
        try:
            # Save video to temp file
            temp_path = "/tmp/temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(video)
            
            # Open video with OpenCV
            cap = cv2.VideoCapture(temp_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # Get video analysis model
            model = self.model_manager.get_model("video")
            
            # Process settings
            threshold = self._get_threshold(settings)
            sample_rate = settings.get("sample_rate", 1.0) if settings else 1.0  # Frames per second to analyze
            
            # Process the video (sample frames)
            frame_results = []
            timestamps = []
            
            frames_to_sample = max(1, int(fps * duration * sample_rate / 100))
            frame_interval = max(1, int(frame_count / frames_to_sample))
            
            frames_analyzed = 0
            for frame_idx in range(0, frame_count, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert OpenCV BGR to RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Analyze frame
                frame_result = model.predict_frame(pil_img, context)
                
                # Store result with timestamp
                timestamp = frame_idx / fps
                timestamps.append({
                    "time": timestamp,
                    "frame_idx": frame_idx,
                    "categories": frame_result
                })
                
                frame_results.append(frame_result)
                frames_analyzed += 1
            
            # Aggregate results across frames
            aggregated_results = self._aggregate_video_results(frame_results)
            
            # Determine if content is harmful based on thresholds
            is_harmful, max_category, confidence = self._evaluate_harmfulness(aggregated_results, threshold)
            
            # Generate explanation
            explanation = self._generate_video_explanation(is_harmful, max_category, confidence, 
                                                          aggregated_results, timestamps)
            
            processing_time = time.time() - start_time
            
            result = {
                "is_harmful": is_harmful,
                "categories": aggregated_results,
                "frames_analyzed": frames_analyzed,
                "timestamps": timestamps,
                "explanation": explanation,
                "confidence": confidence,
                "processing_time": processing_time
            }
            
            logger.info(f"Video processing completed in {processing_time:.4f}s. Harmful: {is_harmful}")
            return result
            
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
            raise
    
    def process_multimodal(self, text: Optional[str] = None, image: Optional[bytes] = None, 
                          video: Optional[bytes] = None, context: Optional[Dict[str, Any]] = None, 
                          settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process multiple content types together, considering their relationships
        
        Args:
            text: Optional text to analyze
            image: Optional image bytes to analyze
            video: Optional video bytes to analyze
            context: Optional contextual information
            settings: Optional processing settings
            
        Returns:
            Dictionary with moderation results
        """
        start_time = time.time()
        logger.info("Processing multimodal content")
        
        try:
            results = {}
            
            # Process each modality separately
            if text:
                results["text"] = self.process_text(text, context, settings)
            
            if image:
                results["image"] = self.process_image(image, context, settings)
                
            if video:
                results["video"] = self.process_video(video, context, settings)
            
            # Get multimodal fusion model
            fusion_model = self.model_manager.get_model("multimodal")
            
            # Fuse the results
            fused_result = fusion_model.fuse_results(results, context)
            
            processing_time = time.time() - start_time
            fused_result["processing_time"] = processing_time
            
            logger.info(f"Multimodal processing completed in {processing_time:.4f}s. Harmful: {fused_result['is_harmful']}")
            return fused_result
            
        except Exception as e:
            logger.error(f"Error in multimodal processing: {str(e)}")
            raise
    
    def _get_threshold(self, settings: Optional[Dict[str, Any]] = None) -> float:
        """Get threshold from settings or use default"""
        return settings.get("threshold", 0.7) if settings else 0.7
    
    def _evaluate_harmfulness(self, categories_result: Dict[str, float], 
                             threshold: float) -> Tuple[bool, str, float]:
        """Determine if content is harmful based on category scores"""
        # Find max category and score
        max_category = max(categories_result.items(), key=lambda x: x[1])
        category_name, confidence = max_category
        
        # Check if any category exceeds threshold
        is_harmful = confidence >= threshold
        
        return is_harmful, category_name, confidence
    
    def _generate_explanation(self, is_harmful: bool, max_category: str, 
                             confidence: float, categories: Dict[str, float]) -> str:
        """Generate human-readable explanation of moderation decision"""
        if not is_harmful:
            return "Content appears to be safe for viewing."
        
        category_desc = self.categories.get(max_category, max_category.replace("_", " ").title())
        explanation = f"Content flagged as potentially harmful ({max_category}) with {confidence:.1%} confidence. {category_desc}."
        
        # Add secondary categories if they have significant scores
        secondary = [(k, v) for k, v in categories.items() 
                     if v >= 0.3 and k != max_category]
        
        if secondary:
            secondary.sort(key=lambda x: x[1], reverse=True)
            secondary_desc = ", ".join([f"{k} ({v:.1%})" for k, v in secondary[:2]])
            explanation += f" Secondary concerns: {secondary_desc}."
            
        return explanation
    
    def _generate_video_explanation(self, is_harmful: bool, max_category: str, confidence: float,
                                  categories: Dict[str, float], timestamps: List[Dict[str, Any]]) -> str:
        """Generate explanation for video content with timestamps"""
        base_explanation = self._generate_explanation(is_harmful, max_category, confidence, categories)
        
        if not is_harmful:
            return base_explanation
            
        # Find most concerning timestamps
        concerning_timestamps = []
        for ts in timestamps:
            ts_categories = ts["categories"]
            max_ts_category = max(ts_categories.items(), key=lambda x: x[1])
            cat_name, score = max_ts_category
            
            if score >= 0.7:  # Use higher threshold for flagging individual frames
                concerning_timestamps.append((ts["time"], cat_name, score))
        
        # Add timestamp information if available
        if concerning_timestamps:
            concerning_timestamps.sort(key=lambda x: x[2], reverse=True)  # Sort by score
            timestamp_info = []
            
            for time, cat, score in concerning_timestamps[:3]:  # Show top 3 concerning timestamps
                mins = int(time // 60)
                secs = int(time % 60)
                timestamp_info.append(f"{mins:02d}:{secs:02d} ({cat}, {score:.1%})")
            
            timestamp_str = ", ".join(timestamp_info)
            base_explanation += f" Concerning content at timestamps: {timestamp_str}."
            
        return base_explanation
    
    def _aggregate_video_results(self, frame_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate results from multiple video frames"""
        if not frame_results:
            return {}
            
        # Initialize with all categories
        aggregated = {cat: 0.0 for cat in self.categories}
        
        # Update with actual values from frames
        for frame_result in frame_results:
            for category, score in frame_result.items():
                if category in aggregated:
                    # Use max pooling (take the highest score across all frames)
                    aggregated[category] = max(aggregated[category], score)
        
        return aggregated 