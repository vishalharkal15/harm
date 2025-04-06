"""
Utility script to download pretrained models for content moderation
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Any, Optional

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import setup_logger
from src.config.model_config import MODEL_CONFIG, SHARED_CONFIG

logger = setup_logger("model_downloader")

def download_model(model_type: str, model_path: str, output_dir: str) -> bool:
    """
    Download a specific model
    
    Args:
        model_type: Type of model (text, image, video)
        model_path: Model identifier (HuggingFace model ID or URL)
        output_dir: Directory to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {model_type} model from {model_path}")
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # In a real implementation, this would use libraries like huggingface_hub
        # to download the models. Here we just simulate the download.
        
        if model_type == "text":
            # Simulate downloading text model from HuggingFace
            logger.info("Downloading tokenizer...")
            time.sleep(1)  # Simulate download
            logger.info("Downloading model weights...")
            time.sleep(2)  # Simulate download
            
            # Create a placeholder file to indicate model is downloaded
            with open(os.path.join(output_dir, f"{model_type}_model_info.txt"), "w") as f:
                f.write(f"Model: {model_path}\n")
                f.write(f"Type: {model_type}\n")
                f.write(f"Downloaded: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
        elif model_type == "image":
            # Simulate downloading image model
            logger.info("Downloading image model weights...")
            time.sleep(2)  # Simulate download
            
            # Create a placeholder file to indicate model is downloaded
            with open(os.path.join(output_dir, f"{model_type}_model_info.txt"), "w") as f:
                f.write(f"Model: {model_path}\n")
                f.write(f"Type: {model_type}\n")
                f.write(f"Downloaded: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
        elif model_type == "video":
            # Simulate downloading video model
            logger.info("Downloading frame analysis model...")
            time.sleep(1.5)  # Simulate download
            logger.info("Downloading temporal model...")
            time.sleep(1.5)  # Simulate download
            
            # Create a placeholder file to indicate model is downloaded
            with open(os.path.join(output_dir, f"{model_type}_model_info.txt"), "w") as f:
                f.write(f"Model: {model_path}\n")
                f.write(f"Type: {model_type}\n")
                f.write(f"Downloaded: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        elif model_type == "multimodal":
            # Simulate downloading multimodal fusion model
            logger.info("Downloading multimodal fusion model...")
            time.sleep(1)  # Simulate download
            
            # Create a placeholder file to indicate model is downloaded
            with open(os.path.join(output_dir, f"{model_type}_model_info.txt"), "w") as f:
                f.write(f"Model: {model_path}\n")
                f.write(f"Type: {model_type}\n")
                f.write(f"Downloaded: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
        else:
            logger.error(f"Unknown model type: {model_type}")
            return False
            
        logger.info(f"Successfully downloaded {model_type} model to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {model_type} model: {str(e)}")
        return False


def download_all_models(output_dir: str, models: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Download all models specified in the configuration
    
    Args:
        output_dir: Base directory to save models
        models: List of model types to download (None for all)
        
    Returns:
        Dictionary with status of each model download
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if models is None:
        models = ["text", "image", "video", "multimodal"]
        
    results = {}
    for model_type in models:
        if model_type in MODEL_CONFIG:
            model_config = MODEL_CONFIG[model_type]
            model_path = model_config.get("path", "")
            model_dir = os.path.join(output_dir, model_type)
            
            success = download_model(model_type, model_path, model_dir)
            results[model_type] = success
        else:
            logger.warning(f"No configuration found for model type: {model_type}")
            results[model_type] = False
            
    return results


def main():
    """Main entry point for the model downloader script"""
    parser = argparse.ArgumentParser(description="Download pretrained models for content moderation")
    parser.add_argument("--output-dir", type=str, default=SHARED_CONFIG.get("cache_dir", "/tmp/model_cache"),
                        help="Directory to save the downloaded models")
    parser.add_argument("--models", type=str, nargs="+", choices=["text", "image", "video", "multimodal"],
                        help="Specific models to download (default: all)")
    
    args = parser.parse_args()
    
    logger.info("Starting model download")
    results = download_all_models(args.output_dir, args.models)
    
    # Print summary
    logger.info("Download summary:")
    for model_type, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {model_type}: {status}")
        
    # Exit with error code if any download failed
    if not all(results.values()):
        logger.error("One or more model downloads failed")
        sys.exit(1)
        
    logger.info("All models downloaded successfully")
    

if __name__ == "__main__":
    main() 