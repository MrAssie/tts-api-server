"""
FastAPI TTS Server
Supports multiple TTS models: XTTS v2, MeloTTS, and KugelAudio
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


def load_xtts_model():
    """Load XTTS v2 model for voice cloning."""
    from TTS.api import TTS
    
    print("Loading XTTS v2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load XTTS v2 model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    print(f"XTTS v2 loaded on {device}")
    return {
        "tts": tts,
        "device": device,
        "name": "XTTS v2",
        "supports_voice_cloning": True,
        "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"]
    }


def load_melotts_model():
    """Load MeloTTS model for fast inference."""
    from melo.api import TTS as MeloTTS
    
    print("Loading MeloTTS model...")
    device = "auto"  # MeloTTS handles device selection
    
    # Load MeloTTS - Dutch model
    model = MeloTTS(language="NL", device=device)
    
    print(f"MeloTTS loaded on {device}")
    return {
        "tts": model,
        "device": device,
        "name": "MeloTTS",
        "supports_voice_cloning": False,
        "languages": ["NL", "EN", "ES", "FR", "ZH", "JP", "KR"]
    }


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
    
    # Try to load XTTS model
    try:
        models["xtts"] = load_xtts_model()
        print("✓ XTTS v2 loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load XTTS v2: {e}")
        traceback.print_exc()
    
    # Try to load MeloTTS model
    try:
        models["melo"] = load_melotts_model()
        print("✓ MeloTTS loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load MeloTTS: {e}")
        traceback.print_exc()
    
    # Try to load KugelAudio model (optional - large model)
    try:
        models["kugel"] = load_kugelaudio_model()
        print("✓ KugelAudio 7B loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load KugelAudio (optional): {e}")
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
class XTTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=10000)
    language: str = Field(default="nl", description="Language code (e.g., nl, en, de)")
    speaker_wav: Optional[str] = Field(default=None, description="Path to reference audio for voice cloning")


class MeloTTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=10000)
    language: str = Field(default="NL", description="Language code (NL, EN, ES, FR, ZH, JP, KR)")
    speaker_id: int = Field(default=0, description="Speaker ID (0-9)")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")


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


@app.post("/tts/xtts")
async def tts_xtts(request: XTTSRequest):
    """
    Generate speech using XTTS v2.
    Supports voice cloning when speaker_wav is provided.
    """
    if "xtts" not in models:
        raise HTTPException(status_code=503, detail="XTTS model not loaded")
    
    try:
        model = models["xtts"]["tts"]
        device = models["xtts"]["device"]
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name
        
        # Generate speech
        if request.speaker_wav and os.path.exists(request.speaker_wav):
            # Voice cloning mode
            model.tts_to_file(
                text=request.text,
                speaker_wav=request.speaker_wav,
                language=request.language,
                file_path=output_path
            )
        else:
            # Default speaker (no cloning)
            # XTTS requires a speaker_wav, use a default or raise error
            raise HTTPException(
                status_code=400, 
                detail="XTTS requires speaker_wav for voice cloning. Please provide a valid audio file path."
            )
        
        # Return audio file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"xtts_output_{hash(request.text)}.wav"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.post("/tts/melo")
async def tts_melo(request: MeloTTSRequest):
    """
    Generate speech using MeloTTS.
    Fast inference, good for Dutch (NL).
    """
    if "melo" not in models:
        raise HTTPException(status_code=503, detail="MeloTTS model not loaded")
    
    try:
        model = models["melo"]["tts"]
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name
        
        # Generate speech
        model.tts_to_file(
            text=request.text,
            speaker_id=request.speaker_id,
            output_path=output_path,
            sdp_ratio=0.2,
            noise_scale=0.6,
            noise_scale_w=0.8,
            speed=request.speed,
            pbar=None  # Disable progress bar
        )
        
        # Return audio file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"melo_output_{hash(request.text)}.wav"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.post("/tts/kugel")
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
