# Troubleshooting Guide

## Common Installation Issues

### Python Version Compatibility

**Problem**: `ERROR: Could not find a version that satisfies the requirement TTS>=0.22.0`

**Solution**: 
- Use Python 3.9-3.11 (avoid 3.12+ for now)
- Use our safe installer: `python install.py`
- Or install minimal dependencies: `pip install -r requirements-minimal.txt`

### PyAudio Installation Issues

**Problem**: `Failed building wheel for pyaudio`

**Solutions**:

**Windows:**
```bash
# Option 1: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Option 2: Use pre-compiled wheel
pip install pipwin
pipwin install pyaudio

# Option 3: Download wheel manually
# From: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
pip install PyAudio-0.2.11-cp39-cp39-win_amd64.whl
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

### FFmpeg Not Found

**Problem**: `ffmpeg not found`

**Solutions**:

**Windows:**
1. Download from https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your PATH

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### Whisper Model Download Issues

**Problem**: Slow or failed Whisper model downloads

**Solutions**:
```bash
# Pre-download models manually
python -c "import whisper; whisper.load_model('base')"

# Or use smaller model
# Edit config.yaml: model: "tiny"
```

## Runtime Issues

### No Audio Input Detected

**Problem**: System doesn't detect microphone input

**Solutions**:
1. Check microphone permissions
2. Test microphone in system settings
3. List audio devices:
```python
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"{i}: {info['name']} - {info['maxInputChannels']} inputs")
```
4. Set specific device in `config.yaml`:
```yaml
audio:
  input_device: 1  # Use device index from above
```

### Speech Recognition Not Working

**Problem**: No text output from speech

**Solutions**:
1. Check microphone levels
2. Speak clearly and loudly
3. Reduce silence threshold in `config.yaml`:
```yaml
audio:
  silence_threshold: 0.005  # Lower = more sensitive
```
4. Try different Whisper model:
```yaml
speech_recognition:
  model: "small"  # or "medium", "large"
```

### Text-to-Speech Not Working

**Problem**: No audio output

**Solutions**:
1. Check system TTS:
   - **Windows**: Should work automatically with SAPI
   - **macOS**: Test with `say "hello"` in terminal
   - **Linux**: Install espeak: `sudo apt-get install espeak`

2. Test audio output:
```python
import pyaudio
import numpy as np

# Generate test tone
sample_rate = 44100
frequency = 440
duration = 1
t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.5 * np.sin(2 * np.pi * frequency * t)

# Play audio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)
stream.write(audio.astype(np.float32).tobytes())
stream.close()
p.terminate()
```

### Memory Issues

**Problem**: System runs out of memory

**Solutions**:
1. Use smaller models in `config.yaml`:
```yaml
speech_recognition:
  model: "tiny"  # Instead of "base" or larger
```

2. Reduce conversation history:
```yaml
memory:
  max_history_length: 10  # Instead of 50
```

3. Disable conversation saving:
```yaml
memory:
  save_conversations: false
```

### LLM Not Available

**Problem**: "LLM not available, using template-based responses"

**Solutions**:
1. Install Ollama: https://ollama.ai
2. Download model: `ollama pull llama3.1:8b`
3. Start Ollama service
4. Or use Hugging Face models (edit `config.yaml`):
```yaml
response_generation:
  llm_provider: "huggingface"
  model_name: "microsoft/DialoGPT-medium"
```

## Performance Issues

### Slow Response Times

**Solutions**:
1. Use faster models:
```yaml
speech_recognition:
  model: "tiny"  # Fastest Whisper model
```

2. Use Faster-Whisper:
```bash
pip install faster-whisper
```

3. Reduce audio processing:
```yaml
audio:
  chunk_size: 2048  # Larger chunks
  silence_duration: 1.0  # Shorter wait time
```

### High CPU Usage

**Solutions**:
1. Use smaller models
2. Increase processing intervals in `main.py`
3. Disable emotion-adaptive TTS:
```yaml
text_to_speech:
  emotion_adaptive: false
```

## Web Interface Issues

### Web Interface Won't Start

**Problem**: `python app.py` fails

**Solutions**:
1. Check if port is in use:
```bash
# Change port in config.yaml
web:
  port: 8001  # Instead of 8000
```

2. Install web dependencies:
```bash
pip install fastapi uvicorn websockets jinja2
```

### WebSocket Connection Failed

**Solutions**:
1. Check firewall settings
2. Try different browser
3. Check browser console for errors

## Getting Help

If you're still having issues:

1. **Check the logs**: Look in `logs/empathic_voice.log`
2. **Run tests**: `python test_system.py`
3. **Minimal test**: Try running individual components
4. **System info**: Include your OS, Python version, and error messages

### Minimal Working Configuration

If nothing works, try this minimal `config.yaml`:

```yaml
audio:
  sample_rate: 16000
  silence_threshold: 0.01

speech_recognition:
  model: "tiny"

emotion_detection:
  enabled: true
  confidence_threshold: 0.5

response_generation:
  llm_provider: "template"  # Use templates only
  default_style: "supportive"

text_to_speech:
  provider: "system"  # Use system TTS
  emotion_adaptive: false

memory:
  enabled: true
  save_conversations: false
```

This configuration uses the most basic, compatible settings that should work on most systems.
