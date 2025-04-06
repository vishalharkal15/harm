import logging
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import internal modules
from src.core.content_processor import ContentProcessor
from src.models.model_manager import ModelManager
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger("api")

# Initialize FastAPI app
app = FastAPI(
    title="SafeGuard AI API",
    description="API for AI-driven content moderation",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()

# Initialize content processor
content_processor = ContentProcessor(model_manager)

# Pydantic models for request/response
class TextModerationRequest(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None

class TextModerationResponse(BaseModel):
    is_harmful: bool
    categories: Dict[str, float]
    explanation: str
    confidence: float
    processing_time: float

class ImageModerationRequest(BaseModel):
    context: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None

class ImageModerationResponse(BaseModel):
    is_harmful: bool
    categories: Dict[str, float]
    explanation: str
    confidence: float
    processing_time: float

class VideoModerationRequest(BaseModel):
    context: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None

class VideoModerationResponse(BaseModel):
    is_harmful: bool
    categories: Dict[str, float]
    frames_analyzed: int
    timestamps: List[Dict[str, Any]]
    explanation: str
    confidence: float
    processing_time: float

@app.get("/")
async def root():
    return {"message": "Welcome to SafeGuard AI Content Moderation API"}

@app.get("/health")
async def health_check():
    health_status = model_manager.health_check()
    return {
        "status": "healthy" if health_status["all_healthy"] else "unhealthy",
        "models": health_status["models"],
        "version": "1.0.0",
    }

@app.post("/moderate/text", response_model=TextModerationResponse)
async def moderate_text(request: TextModerationRequest):
    try:
        result = content_processor.process_text(
            text=request.text,
            context=request.context,
            settings=request.settings
        )
        return result
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/moderate/image", response_model=ImageModerationResponse)
async def moderate_image(
    image: UploadFile = File(...),
    context: Optional[str] = Form(None),
    settings: Optional[str] = Form(None)
):
    try:
        result = content_processor.process_image(
            image=await image.read(),
            context=context,
            settings=settings
        )
        return result
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/moderate/video", response_model=VideoModerationResponse)
async def moderate_video(
    video: UploadFile = File(...),
    context: Optional[str] = Form(None),
    settings: Optional[str] = Form(None)
):
    try:
        result = content_processor.process_video(
            video=await video.read(),
            context=context,
            settings=settings
        )
        return result
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/moderate/multimodal")
async def moderate_multimodal(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    context: Optional[str] = Form(None),
    settings: Optional[str] = Form(None)
):
    try:
        result = content_processor.process_multimodal(
            text=text,
            image=await image.read() if image else None,
            video=await video.read() if video else None,
            context=context,
            settings=settings
        )
        return result
    except Exception as e:
        logger.error(f"Error processing multimodal content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True) 