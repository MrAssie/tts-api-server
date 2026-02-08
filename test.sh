#!/bin/bash
# Test script voor TTS API Server

echo "=================================="
echo "TTS API Test Script"
echo "=================================="

BASE_URL="${TTS_API_URL:-http://localhost:8000}"
echo "Testing against: $BASE_URL"
echo ""

# Test 1: Health check
echo "[Test 1] Health check..."
curl -s "$BASE_URL/health" | python3 -m json.tool || echo "Health check failed"
echo ""

# Test 2: List models
echo "[Test 2] List available models..."
curl -s "$BASE_URL/models" | python3 -m json.tool || echo "Models list failed"
echo ""

# Test 3: MeloTTS GET request
echo "[Test 3] MeloTTS (GET - simple)..."
curl -s "$BASE_URL/tts/melo?text=Hallo%20wereld%2C%20dit%20is%20een%20test&language=NL" \
  --output /tmp/test_melo_get.wav

if [ -f /tmp/test_melo_get.wav ] && [ -s /tmp/test_melo_get.wav ]; then
    echo "✓ MeloTTS GET: Audio saved to /tmp/test_melo_get.wav"
    ls -lh /tmp/test_melo_get.wav
else
    echo "✗ MeloTTS GET failed"
fi
echo ""

# Test 4: MeloTTS POST request
echo "[Test 4] MeloTTS (POST with options)..."
curl -s -X POST "$BASE_URL/tts/melo" \
  -H "Content-Type: application/json" \
  -d '{"text": "Dit is een test met MeloTTS in het Nederlands", "language": "NL", "speaker_id": 0, "speed": 1.0}' \
  --output /tmp/test_melo_post.wav

if [ -f /tmp/test_melo_post.wav ] && [ -s /tmp/test_melo_post.wav ]; then
    echo "✓ MeloTTS POST: Audio saved to /tmp/test_melo_post.wav"
    ls -lh /tmp/test_melo_post.wav
else
    echo "✗ MeloTTS POST failed"
fi
echo ""

# Test 5: XTTS (voorbeeld - vereist speaker_wav)
echo "[Test 5] XTTS info..."
echo "XTTS vereist een speaker_wav bestand voor voice cloning."
echo "Voorbeeld commando:"
echo "  curl -X POST $BASE_URL/tts/xtts \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"text\": \"Hallo\", \"language\": \"nl\", \"speaker_wav\": \"/path/to/voice.wav\"}' \\"
echo "    --output output.wav"
echo ""

echo "=================================="
echo "Test complete!"
echo "=================================="
