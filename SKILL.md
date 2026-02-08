# TTS API Server

FastAPI-based Text-to-Speech server met ondersteuning voor meerdere modellen, geoptimaliseerd voor GPU (RTX 5060 Ti).

## Ondersteunde Modellen

| Model | Beste voor | Voice Cloning | Talen | Snelheid | Params |
|-------|-----------|---------------|-------|----------|--------|
| XTTS v2 | Hoge kwaliteit | ✅ Ja | 16+ talen | Medium | 400M |
| MeloTTS | Snel NL/EN | ❌ Nee | 7 talen | Snel | 100M |
| **KugelAudio** | **Beste EU talen** | ✅ Ja | European | Langzaam | **7B** |

## Snel Starten

### Lokale Installatie

```bash
# 1. Ga naar de server directory
cd server

# 2. Maak virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# of: venv\Scripts\activate  # Windows

# 3. Installeer PyTorch met CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Installeer overige dependencies
pip install -r requirements.txt

# 5. Download unidic voor MeloTTS
python -m unidic download

# 6. Start de server
python main.py
```

### Docker (Aanbevolen)

```bash
# 1. Bouw de image
cd server
docker build -t tts-api .

# 2. Start met GPU support
docker run -d \
  --name tts-api \
  --gpus all \
  -p 8000:8000 \
  -v tts-cache:/app/.cache \
  tts-api
```

## API Endpoints

### GET /health
Status check van de server.

**Response:**
```json
{
  "status": "healthy",
  "cuda_available": true,
  "cuda_device": "NVIDIA RTX 5060 Ti",
  "loaded_models": ["xtts", "melo"]
}
```

### GET /models
Lijst van beschikbare modellen.

**Response:**
```json
{
  "models": [
    {
      "id": "xtts",
      "name": "XTTS v2",
      "supports_voice_cloning": true,
      "languages": ["en", "es", "fr", "de", "nl", ...]
    },
    {
      "id": "melo",
      "name": "MeloTTS",
      "supports_voice_cloning": false,
      "languages": ["NL", "EN", "ES", "FR", "ZH", "JP", "KR"]
    },
    {
      "id": "kugel",
      "name": "KugelAudio 7B",
      "supports_voice_cloning": true,
      "languages": ["de", "en", "fr", "es", "pl", "it", "nl"]
    }
  ]
}
```

### POST /tts/xtts
XTTS v2 tekst-naar-spraak met voice cloning.

**Request:**
```json
{
  "text": "Hallo, dit is een test in het Nederlands.",
  "language": "nl",
  "speaker_wav": "/path/to/reference.wav"
}
```

**Response:** Audio file (WAV)

### POST /tts/melo
MeloTTS snelle tekst-naar-spraak.

**Request:**
```json
{
  "text": "Hallo, dit is een test in het Nederlands.",
  "language": "NL",
  "speaker_id": 0,
  "speed": 1.0
}
```

**Response:** Audio file (WAV)

### POST /tts/kugel
KugelAudio 7B - State-of-the-art voor Europese talen.

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

**Note:** KugelAudio vereist ~14GB VRAM. Zorg dat je RTX 5060 Ti 16GB gebruikt.

## Voorbeelden

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Lijst modellen
curl http://localhost:8000/models

# MeloTTS (simpel - GET)
curl "http://localhost:8000/tts/melo?text=Hallo%20wereld&language=NL" \
  --output output.wav

# MeloTTS (POST met opties)
curl -X POST http://localhost:8000/tts/melo \
  -H "Content-Type: application/json" \
  -d '{"text": "Hallo wereld", "language": "NL", "speaker_id": 0, "speed": 1.2}' \
  --output output.wav

# XTTS met voice cloning
curl -X POST http://localhost:8000/tts/xtts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hallo, dit is mijn stem", "language": "nl", "speaker_wav": "/path/to/voice.wav"}' \
  --output cloned.wav
```

### Python

```python
import requests

# MeloTTS
response = requests.post(
    "http://localhost:8000/tts/melo",
    json={"text": "Hallo wereld", "language": "NL"}
)
with open("output.wav", "wb") as f:
    f.write(response.content)

# XTTS met voice cloning
response = requests.post(
    "http://localhost:8000/tts/xtts",
    json={
        "text": "Hallo, dit is mijn gekloonde stem",
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

Starten:
```bash
docker-compose up -d
```

## Model Details

### XTTS v2
- **Beste voor:** Hoge kwaliteit, voice cloning
- **Requirements:** Referentie audio (3-10 sec) voor voice cloning
- **Talen:** 16+ inclusief Nederlands (nl), Engels (en), Duits (de), etc.
- **GPU Memory:** ~4-6 GB
- **Snelheid:** ~1x real-time op RTX 5060 Ti

### MeloTTS
- **Beste voor:** Snelle inference, Nederlandse tekst
- **Requirements:** Geen referentie audio nodig
- **Talen:** NL, EN, ES, FR, ZH, JP, KR
- **GPU Memory:** ~1-2 GB
- **Snelheid:** ~10x real-time op RTX 5060 Ti

### KugelAudio 7B
- **Beste voor:** Hoogste kwaliteit Europese talen
- **Requirements:** ~14GB VRAM (je 16GB RTX 5060 Ti is perfect!)
- **Talen:** Duits (de), Engels (en), Frans (fr), Spaans (es), Pools (pl), Italiaans (it), Nederlands (nl)
- **GPU Memory:** ~14 GB
- **Snelheid:** ~0.5x real-time (langzamer, maar beste kwaliteit)
- **Voice Cloning:** Ja, met referentie audio
- **Architectuur:** AR + Diffusion (7B parameters)
- **Repo:** https://github.com/Kugelaudio/kugelaudio-open

## Troubleshooting

### CUDA out of memory
```bash
# Verminder batch size of gebruik CPU fallback
# In code: device = "cpu" als CUDA faalt
```

### Modellen laden niet
```bash
# Check CUDA installatie
nvidia-smi

# Herstart met schone cache
rm -rf ~/.cache/tts ~/.cache/huggingface
docker-compose restart
```

### ESpeak niet gevonden (MeloTTS)
```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng

# In Docker: al geïnstalleerd in image
```

## Environment Variables

| Variable | Default | Beschrijving |
|----------|---------|--------------|
| `CUDA_VISIBLE_DEVICES` | `0` | Welke GPU te gebruiken |
| `TORCH_HOME` | `/app/.cache/torch` | PyTorch cache directory |
| `HF_HOME` | `/app/.cache/huggingface` | HuggingFace cache directory |

## Uitbreiden met Nieuwe Modellen

1. Voeg model loader toe in `main.py`:
```python
def load_custom_model():
    # Jouw model loading code
    return {"tts": model, "device": device, ...}
```

2. Registreer in lifespan:
```python
models["custom"] = load_custom_model()
```

3. Voeg endpoint toe:
```python
@app.post("/tts/custom")
async def tts_custom(request: CustomRequest):
    # Jouw inference code
    return FileResponse(output_path)
```

## Links

- [Coqui TTS Docs](https://docs.coqui.ai/en/latest/)
- [MeloTTS GitHub](https://github.com/myshell-ai/MeloTTS)
- [XTTS v2 Paper](https://arxiv.org/abs/2406.04904)
