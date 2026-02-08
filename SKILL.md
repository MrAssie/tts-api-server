# TTS API Server

FastAPI-based Text-to-Speech server met **KugelAudio 7B** - state-of-the-art voor Europese talen.

## Model

| Model | Beste voor | Voice Cloning | Talen | Params | VRAM |
|-------|-----------|---------------|-------|--------|------|
| **KugelAudio 7B** | Beste EU kwaliteit | ✅ Ja | 7 EU talen | 7B | ~14GB |

**Talen:** Duits (de), Engels (en), Frans (fr), Spaans (es), Pools (pl), Italiaans (it), **Nederlands (nl)**

## Snel Starten

### Docker (Aanbevolen)

```bash
# Clone repo
git clone https://github.com/MrAssie/tts-api-server.git
cd tts-api-server

# Start met GPU support
docker-compose up -d
```

### Lokale Installatie

```bash
# 1. Ga naar de server directory
cd server

# 2. Maak virtual environment
python -m venv venv
source venv/bin/activate

# 3. Installeer PyTorch met CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Installeer dependencies
pip install -r requirements.txt

# 5. Start de server
python main.py
```

## API Endpoints

### GET /health
Status check.

**Response:**
```json
{
  "status": "healthy",
  "cuda_available": true,
  "cuda_device": "NVIDIA RTX 5060 Ti",
  "loaded_models": ["kugel"]
}
```

### POST /tts
KugelAudio 7B tekst-naar-spraak.

**Request:**
```json
{
  "text": "Hallo, dit is een test in het Nederlands.",
  "language": "nl",
  "speaker_wav": "/path/to/reference.wav",
  "temperature": 0.8
}
```

**Response:** Audio file (WAV)

### GET /tts (test)
Simpel test endpoint.

```bash
curl "http://localhost:8000/tts?text=Hallo%20wereld&language=nl" \
  --output test.wav
```

## Voorbeelden

### cURL

```bash
# Health check
curl http://localhost:8000/health

# TTS met KugelAudio
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hallo wereld", "language": "nl"}' \
  --output output.wav

# TTS met voice cloning
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Dit is mijn gekloonde stem",
    "language": "nl",
    "speaker_wav": "/path/to/voice.wav",
    "temperature": 0.8
  }' \
  --output cloned.wav
```

### Python

```python
import requests

# TTS
response = requests.post(
    "http://localhost:8000/tts",
    json={"text": "Hallo wereld", "language": "nl"}
)
with open("output.wav", "wb") as f:
    f.write(response.content)

# Voice cloning
response = requests.post(
    "http://localhost:8000/tts",
    json={
        "text": "Hallo, dit is mijn stem",
        "language": "nl",
        "speaker_wav": "/path/to/reference.wav"
    }
)
with open("cloned.wav", "wb") as f:
    f.write(response.content)
```

## Docker Compose

```yaml
version: '3.8'

services:
  tts-api:
    build: ./server
    container_name: tts-api
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - tts-cache:/app/.cache
      - ./voices:/app/voices:ro
    environment:
      - CUDA_VISIBLE_DEVICES=0
    restart: unless-stopped

volumes:
  tts-cache:
```

## Model Details

### KugelAudio 7B
- **Beste voor:** Hoogste kwaliteit Europese talen
- **Requirements:** ~14GB VRAM (je 16GB RTX 5060 Ti is perfect!)
- **Talen:** de, en, fr, es, pl, it, **nl**
- **GPU Memory:** ~14 GB
- **Snelheid:** ~0.5x real-time
- **Voice Cloning:** Ja, met referentie audio
- **Architectuur:** AR + Diffusion (7B parameters)
- **Paper:** State-of-the-art op EmergentTTS-Eval

## Links

- [KugelAudio GitHub](https://github.com/Kugelaudio/kugelaudio-open)
- [HuggingFace Model](https://huggingface.co/kugelaudio/kugelaudio-0-open)
