import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Union
import numpy as np
from PIL import Image
import io

def format_results(results: Dict[str, Any], format_type: str = "json") -> Union[str, Dict[str, Any]]:
    """
    Format moderation results into different output formats
    
    Args:
        results: Dictionary of moderation results
        format_type: Output format (json, html, text)
        
    Returns:
        Formatted results as string or dict depending on format_type
    """
    if format_type == "json":
        return results
    
    elif format_type == "html":
        html = "<div class='moderation-results'>\n"
        
        # Add harmful status
        if results.get("is_harmful", False):
            html += "<div class='result-harmful'>Content flagged as potentially harmful</div>\n"
        else:
            html += "<div class='result-safe'>Content appears to be safe</div>\n"
            
        # Add explanation
        if "explanation" in results:
            html += f"<div class='result-explanation'>{results['explanation']}</div>\n"
            
        # Add categories
        if "categories" in results:
            html += "<div class='result-categories'>\n"
            html += "<h3>Category Scores:</h3>\n"
            html += "<ul>\n"
            
            categories = results["categories"]
            for category, score in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                color = get_color_for_score(score)
                html += f"<li style='color: {color}'>{category}: {score:.1%}</li>\n"
                
            html += "</ul>\n</div>\n"
            
        html += "</div>"
        return html
    
    elif format_type == "text":
        text = []
        
        # Add harmful status
        if results.get("is_harmful", False):
            text.append("CONTENT FLAGGED: Potentially harmful content detected")
        else:
            text.append("CONTENT SAFE: No harmful content detected")
            
        # Add explanation
        if "explanation" in results:
            text.append(f"\nExplanation: {results['explanation']}")
            
        # Add categories
        if "categories" in results:
            text.append("\nCategory Scores:")
            
            categories = results["categories"]
            for category, score in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                text.append(f"- {category}: {score:.1%}")
                
        # Add processing time
        if "processing_time" in results:
            text.append(f"\nProcessing time: {results['processing_time']:.2f} seconds")
            
        return "\n".join(text)
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def get_color_for_score(score: float) -> str:
    """
    Get a color representing the severity of a score (for visualization)
    
    Args:
        score: Score value between 0 and 1
        
    Returns:
        CSS-compatible color string
    """
    if score < 0.3:
        return "#1a9641"  # Green
    elif score < 0.6:
        return "#fdae61"  # Yellow/Orange
    else:
        return "#d7191c"  # Red


def compute_content_hash(content: Union[str, bytes, Image.Image]) -> str:
    """
    Compute a hash for content to enable caching and deduplication
    
    Args:
        content: Content to hash (text, image bytes, or PIL Image)
        
    Returns:
        Hash string
    """
    hasher = hashlib.sha256()
    
    if isinstance(content, str):
        hasher.update(content.encode('utf-8'))
    elif isinstance(content, bytes):
        hasher.update(content)
    elif isinstance(content, Image.Image):
        img_bytes = io.BytesIO()
        content.save(img_bytes, format='PNG')
        hasher.update(img_bytes.getvalue())
    else:
        raise TypeError(f"Unsupported content type: {type(content)}")
        
    return hasher.hexdigest()


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a list of items into batches for efficient processing
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches, where each batch is a list of items
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def estimate_model_size(model_dict: Dict[str, Any]) -> float:
    """
    Estimate the size of a model in MB based on its parameters
    
    Args:
        model_dict: Dictionary with model parameters
        
    Returns:
        Estimated size in MB
    """
    total_params = 0
    for name, tensor in model_dict.items():
        if hasattr(tensor, 'shape'):
            total_params += np.prod(tensor.shape)
    
    # Assuming 32-bit floats (4 bytes per parameter)
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb


def sanitize_text(text: str) -> str:
    """
    Sanitize text input to remove potential security issues
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
        
    # Remove common XSS patterns
    text = text.replace('<script>', '')
    text = text.replace('</script>', '')
    text = text.replace('javascript:', '')
    
    # Limit length
    max_length = 10000  # Adjust as needed
    if len(text) > max_length:
        text = text[:max_length]
        
    return text


def download_models(model_dir: str, models: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Download pretrained models (placeholder for actual implementation)
    
    Args:
        model_dir: Directory to save models
        models: List of model names to download (None for all)
        
    Returns:
        Dictionary with status of each model download
    """
    # This would be implemented to actually download models in a real system
    # Here we just simulate the functionality
    
    available_models = {
        "text": "distilbert-base-uncased-finetuned-sst-2-english",
        "image": "efficientnet-b0",
        "video": "video_model"
    }
    
    if models is None:
        models = list(available_models.keys())
        
    results = {}
    for model in models:
        if model in available_models:
            # Simulate download
            time.sleep(0.5)
            results[model] = True
        else:
            results[model] = False
            
    return results 