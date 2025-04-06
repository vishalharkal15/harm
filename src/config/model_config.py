"""
Configuration for AI models used in content moderation
"""

MODEL_CONFIG = {
    "text": {
        "path": "distilbert-base-uncased-finetuned-sst-2-english",  # Placeholder - would use actual fine-tuned model
        "batch_size": 32,
        "max_length": 512,
        "threshold": 0.7,
        "categories": [
            "hate_speech",
            "harassment", 
            "violence",
            "self_harm",
            "sexual",
            "child_exploitation",
            "terrorism",
            "misinformation",
            "spam",
            "illegal_activity"
        ],
        "device": "cuda",  # Use GPU if available
        "precision": "fp16",  # Mixed precision for faster inference
        "cache_dir": "/tmp/model_cache/text"
    },
    
    "image": {
        "path": "efficientnet-b0",  # Placeholder - would use actual fine-tuned model
        "input_size": [224, 224],
        "batch_size": 16,
        "threshold": 0.7,
        "categories": [
            "hate_speech",
            "violence",
            "self_harm",
            "sexual",
            "child_exploitation",
            "terrorism"
        ],
        "device": "cuda",
        "precision": "fp16",
        "cache_dir": "/tmp/model_cache/image"
    },
    
    "video": {
        "path": "video_model",  # Placeholder - would use actual model path
        "frame_sample_rate": 1.0,  # Sample 1 frame per second
        "batch_size": 8,
        "threshold": 0.7,
        "categories": [
            "hate_speech",
            "violence",
            "self_harm",
            "sexual",
            "child_exploitation",
            "terrorism"
        ],
        "temporal_window": 5,  # Number of frames to consider for temporal context
        "device": "cuda",
        "precision": "fp16",
        "cache_dir": "/tmp/model_cache/video"
    },
    
    "multimodal": {
        "fusion_strategy": "attention",  # Options: simple, weighted, attention
        "threshold": 0.6,  # Lower threshold for multimodal since we have more evidence
        "device": "cuda",
        "cache_dir": "/tmp/model_cache/multimodal"
    }
}

# Shared settings across all models
SHARED_CONFIG = {
    "use_cache": True,
    "cache_dir": "/tmp/model_cache",
    "log_level": "INFO",
    "fallback_to_cpu": True,  # If GPU not available, fall back to CPU
    "precision": "fp32",  # Default precision
    "trusted_model_sources": [
        "huggingface.co",
        "pytorch.org"
    ]
}

# Ethical considerations
ETHICAL_CONFIG = {
    "bias_mitigation": {
        "enabled": True,
        "demographic_parity": True,
        "equal_opportunity": True
    },
    "explainability": {
        "enabled": True,
        "method": "attention",  # Options: attention, lime, shap
        "granularity": "medium"  # Options: low, medium, high
    },
    "human_review_threshold": 0.4,  # Scores between 0.4-0.7 might warrant human review
    "content_storage": {
        "store_sensitive_content": False,
        "retention_period_days": 30,
        "encryption": "AES-256"
    }
}

# Performance settings
PERFORMANCE_CONFIG = {
    "batch_inference": True,
    "parallel_processing": True,
    "num_workers": 4,
    "prefetch_factor": 2,
    "timeout_seconds": 30,
    "max_retries": 3
}

# System limits to prevent abuse
SYSTEM_LIMITS = {
    "max_text_length": 50000,  # characters
    "max_image_size": 10 * 1024 * 1024,  # 10MB
    "max_video_size": 100 * 1024 * 1024,  # 100MB
    "max_video_duration": 300,  # 5 minutes
    "max_requests_per_minute": 60,
    "max_items_per_request": 10
} 