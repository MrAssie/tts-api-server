# TTS API Server

FastAPI-based Text-to-Speech server met **KugelAudio 7B** - officiële implementatie.

## Model

| Model | Beste voor | Stemmen | Talen | Params | VRAM |
|-------|-----------|---------|-------|--------|------|
| **KugelAudio 7B** | Beste EU kwaliteit | default, warm, clear | 23 EU talen | 7B | ~19GB |

**Talen:** English (en), German (de), French (fr), Spanish (es), Italian (it), Portuguese (pt), **Dutch (nl)**, Polish (pl), Russian (ru), Ukrainian (uk), Czech (cs), Romanian (ro), Hungarian (hu), Swedish (sv), Danish (da), Finnish (fi), Norwegian (no), Greek (el), Bulgarian (bg), Slovak (sk), Croatian (hr), Serbian (sr), Turkish (tr)

## Snel Starten

### Docker (Aanbevolen)

```bash
# Clone repo
git clone https://github.com/MrAssie/tts-api-server.git
cd tts-api-server

# Start met GPU support
docker-compose up -d
```

### Lokale Installatie (met uv)

```bash
# 1. Clone en ga naar server directory
git clone https://github.com/MrAssie/tts-api-server.git
cd tts-api-server/server

# 2. Installeer uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Installeer dependencies
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install kugelaudio-open
pip install -r requirements.txt

# 4. Start de server
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

### GET /voices
Lijst beschikbare stemmen.

**Response:**
```json
{
  "voices": ["default", "warm", "clear"],
  "total": 3
}
```

### POST /tts
KugelAudio 7B tekst-naar-spraak.

**Request:**
```json
{
  "text": "Hallo, dit is een test in het Nederlands.",
  "voice": "default",
  "language": "nl",
  "cfg_scale": 3.0
}
```

**Parameters:**
- `text` (string, verplicht): Tekst om om te zetten
- `voice` (string, optioneel): Stem naam (`default`, `warm`, `clear`)
- `language` (string, optioneel): Taal code (bijv. `nl`, `en`, `de`)
- `cfg_scale` (float, optioneel): Guidance scale (1.0-10.0, default: 3.0)

**Response:** Audio file (WAV)

### GET /tts (test)
Simpel test endpoint.

```bash
curl "http://localhost:8000/tts?text=Hallo%20wereld&voice=warm" \
  --output test.wav
```

## Voorbeelden

### cURL

```bash
# Health check
curl http://localhost:8080/health

# Lijst stemmen
curl http://localhost:8080/voices

# TTS met default stem
curl -X POST http://localhost:8080/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hallo wereld", "voice": "default"}' \
  --output output.wav

# TTS met warme stem
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welkom bij KugelAudio",
    "voice": "warm",
    "cfg_scale": 3.0
  }' \
  --output warm.wav
```

### Python

```python
import requests

# TTS
response = requests.post(
    "http://localhost:8080/tts",
    json={
        "text": "Hallo wereld",
        "voice": "default",
        "language": "nl"
    }
)
with open("output.wav", "wb") as f:
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
      - "8080:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - tts-cache:/app/.cache
    environment:
      - CUDA_VISIBLE_DEVICES=0
    restart: unless-stopped

volumes:
  tts-cache:
```

## Model Details

### KugelAudio 7B
- **Beste voor:** Hoogste kwaliteit Europese talen
- **Requirements:** ~19GB VRAM (je 16GB RTX 5060 Ti is net genoeg!)
- **Stemmen:** `default`, `warm`, `clear`
- **Talen:** 23 Europese talen
- **GPU Memory:** ~19 GB
- **Snelheid:** ~1x real-time
- **Architectuur:** AR + Diffusion (7B parameters)
- **Official Repo:** https://github.com/Kugelaudio/kugelaudio-open

### Stemmen

- **default**: Standaard neutrale stem
- **warm**: Warme, vriendelijke stem
- **clear**: Heldere, duidelijke stem

## Links

- [KugelAudio GitHub](https://github.com/Kugelaudio/kugelaudio-open)
- [HuggingFace Model](https://huggingface.co/kugelaudio/kugelaudio-0-open)
- [KugelAudio Website](https://kugelaudio.com)
