"""
FastAPI TTS Server - KugelAudio 7B
Official implementation using kugelaudio-open package
Optimized for NVIDIA GPU (RTX 5060 Ti)
"""

import os
import tempfile
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Global model cache
models = {}


def load_kugelaudio_model():
    """Load KugelAudio 7B model using official package."""
    from kugelaudio_open import (
        KugelAudioForConditionalGenerationInference,
        KugelAudioProcessor,
    )
    
    print("Loading KugelAudio 7B model...")
    print("This may take a while (7B parameters)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "kugelaudio/kugelaudio-0-open"
    
    # Load model - use device_map="auto" for better memory management
    model = KugelAudioForConditionalGenerationInference.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Let transformers handle device placement
    )
    model.eval()
    
    # Strip encoder weights to save VRAM (only decoders needed for inference)
    model.model.strip_encoders()
    
    # Load processor
    processor = KugelAudioProcessor.from_pretrained(model_name)
    
    print(f"KugelAudio loaded on {device}")
    print(f"Available voices: {processor.get_available_voices()}")
    
    return {
        "model": model,
        "processor": processor,
        "device": device,
        "name": "KugelAudio 7B",
        "languages": ["en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "uk", "cs", "ro", "hu", "sv", "da", "fi", "no", "el", "bg", "sk", "hr", "sr", "tr"]
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    print("=" * 50)
    print("Starting TTS Server...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load KugelAudio model
    try:
        models["kugel"] = load_kugelaudio_model()
        print("✓ KugelAudio 7B loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load KugelAudio: {e}")
        traceback.print_exc()
    
    print("=" * 50)
    yield
    
    # Cleanup
    print("Shutting down TTS Server...")
    models.clear()


app = FastAPI(
    title="KugelAudio TTS Server",
    description="Text-to-Speech API using KugelAudio 7B",
    version="1.0.0",
    lifespan=lifespan
)


# Pydantic models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=5000)
    voice: str = Field(default="default", description="Voice to use (default, warm, clear)")
    language: str = Field(default="nl", description="Language code")
    cfg_scale: float = Field(default=3.0, ge=1.0, le=10.0, description="Guidance scale")


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded_models": list(models.keys())
    }


@app.get("/voices")
async def list_voices():
    """List available voices."""
    if "kugel" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    processor = models["kugel"]["processor"]
    voices = processor.get_available_voices()
    
    return {
        "voices": voices,
        "total": len(voices)
    }


@app.post("/tts")
async def generate_tts(request: TTSRequest):
    """
    Generate speech using KugelAudio 7B.
    """
    if "kugel" not in models:
        raise HTTPException(status_code=503, detail="KugelAudio model not loaded")
    
    try:
        model = models["kugel"]["model"]
        processor = models["kugel"]["processor"]
        device = models["kugel"]["device"]
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name
        
        # Process inputs
        inputs = processor(
            text=request.text,
            voice=request.voice,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate speech
        with torch.no_grad():
            outputs = model.generate(**inputs, cfg_scale=request.cfg_scale)
        
        # Save audio
        processor.save_audio(outputs.speech_outputs[0], output_path)
        
        # Return audio file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"kugelaudio_{hash(request.text)}.wav"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.get("/tts")
async def generate_tts_get(
    text: str = Query(..., description="Text to synthesize"),
    voice: str = Query(default="default", description="Voice to use"),
    language: str = Query(default="nl", description="Language code"),
    cfg_scale: float = Query(default=3.0, description="Guidance scale")
):
    """GET endpoint for TTS (for simple testing)."""
    request = TTSRequest(text=text, voice=voice, language=language, cfg_scale=cfg_scale)
    return await generate_tts(request)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
