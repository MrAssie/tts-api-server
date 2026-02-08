"""
FastAPI TTS Server
Supports KugelAudio 7B - High-quality European language TTS
Optimized for GPU acceleration (CUDA)
"""

import os
import io
import tempfile
import traceback
from typing import Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Global model cache
models = {}


def load_kugelaudio_model():
    """Load KugelAudio 7B model for high-quality European language TTS."""
    from transformers import AutoModel, AutoTokenizer
    
    print("Loading KugelAudio 7B model...")
    print("This may take a while (7B parameters)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "kugelaudio/kugelaudio-0-open"
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    if device == "cpu":
        model = model.to(device)
    
    print(f"KugelAudio loaded on {device}")
    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "name": "KugelAudio 7B",
        "supports_voice_cloning": True,
        "languages": ["de", "en", "fr", "es", "pl", "it", "nl"]  # European languages
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
    title="TTS API Server",
    description="Multi-model Text-to-Speech API with GPU acceleration",
    version="1.0.0",
    lifespan=lifespan
)


# Pydantic models
class KugelAudioRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=5000)
    language: str = Field(default="nl", description="Language code (de, en, fr, es, pl, it, nl)")
    speaker_wav: Optional[str] = Field(default=None, description="Path to reference audio for voice cloning")
    temperature: float = Field(default=0.8, ge=0.1, le=1.0, description="Sampling temperature")


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


@app.get("/models")
async def list_models():
    """List available TTS models."""
    available_models = []
    
    for model_id, model_info in models.items():
        available_models.append({
            "id": model_id,
            "name": model_info["name"],
            "device": model_info["device"],
            "supports_voice_cloning": model_info["supports_voice_cloning"],
            "languages": model_info["languages"]
        })
    
    return {
        "models": available_models,
        "total": len(available_models)
    }


@app.post("/tts")
async def tts_kugel(request: KugelAudioRequest):
    """
    Generate speech using KugelAudio 7B.
    High-quality European language TTS with voice cloning.
    """
    if "kugel" not in models:
        raise HTTPException(status_code=503, detail="KugelAudio model not loaded")
    
    try:
        model = models["kugel"]["model"]
        tokenizer = models["kugel"]["tokenizer"]
        device = models["kugel"]["device"]
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name
        
        # Prepare input
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        # Generate audio
        with torch.no_grad():
            # Note: This is a simplified version
            # Actual implementation depends on KugelAudio's specific API
            outputs = model.generate(
                **inputs,
                temperature=request.temperature,
                do_sample=True
            )
        
        # Save audio (implementation depends on model output format)
        # This is placeholder - actual code depends on KugelAudio's output format
        # audio = process_outputs(outputs)
        # save_audio(audio, output_path)
        
        # For now, return a placeholder response
        raise HTTPException(
            status_code=501,
            detail="KugelAudio integration requires specific model implementation. Please check kugelaudio-open repository for exact API usage."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.get("/tts/xtts")
async def tts_xtts_get(
    text: str = Query(..., description="Text to synthesize"),
    language: str = Query(default="nl", description="Language code"),
    speaker_wav: Optional[str] = Query(default=None, description="Path to reference audio")
):
    """GET endpoint for XTTS (for simple testing)."""
    request = XTTSRequest(text=text, language=language, speaker_wav=speaker_wav)
    return await tts_xtts(request)


@app.get("/tts/melo")
async def tts_melo_get(
    text: str = Query(..., description="Text to synthesize"),
    language: str = Query(default="NL", description="Language code"),
    speaker_id: int = Query(default=0, description="Speaker ID"),
    speed: float = Query(default=1.0, description="Speed multiplier")
):
    """GET endpoint for MeloTTS (for simple testing)."""
    request = MeloTTSRequest(text=text, language=language, speaker_id=speaker_id, speed=speed)
    return await tts_melo(request)


if __name__ == "__main__":
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
