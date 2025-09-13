# Getting Started with Empathic Voice Companion

Welcome to the **Empathic Voice Companion** - a unique AI system that detects emotional states from your voice and responds with genuine empathy using cutting-edge open-source tools.

## 🌟 What Makes This Special

- **Real-time Emotion Detection**: Analyzes voice tone, pitch, and prosodic features to understand your emotional state
- **Empathic Response Generation**: Uses advanced AI to craft responses that validate and support your feelings
- **Multiple Interaction Modes**: Voice-to-voice, text-based web interface, and API access
- **Privacy-First**: All processing happens locally - your conversations never leave your device
- **Conversation Memory**: Tracks emotional patterns and adapts to your preferences over time
- **Multiple Empathy Styles**: Supportive, reflective, solution-focused, and therapeutic approaches

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone or download the project
cd empathic-voice-companion

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Use the safe installer (handles compatibility issues)
python install.py

# OR install minimal dependencies manually
pip install -r requirements-minimal.txt
```

### 2. Setup Models and System Dependencies

```bash
# Run the automated setup script
python setup_models.py

# This will:
# - Check system requirements
# - Install audio dependencies (FFmpeg, PyAudio)
# - Download Whisper models for speech recognition
# - Setup Piper TTS for speech synthesis
# - Optionally setup Ollama for advanced LLM responses
```

### 3. Test Your Installation

```bash
# Run system tests
python test_system.py

# This will verify all components are working correctly
```

### 4. Start Using the System

#### Option A: Voice Interface (Recommended)
```bash
python main.py
```
- Speak naturally into your microphone
- The system will detect your emotions and respond empathically
- Press Ctrl+C to stop

#### Option B: Web Interface
```bash
python app.py
```
- Open http://localhost:8000 in your browser
- Type messages and receive empathic responses
- View emotion patterns and conversation history

## 🎯 Usage Examples

### Voice Interaction
```
You: "I'm feeling really overwhelmed with work today"
🎯 Processing speech...
😊 Detected emotion: anxious (confidence: 0.78)
🤖 Response: "I can hear the stress and worry in your voice. Feeling overwhelmed at work is really challenging, and it's completely understandable. You're not alone in feeling this way. What's been the most overwhelming part of your day?"
🔊 Generating speech...
✅ Speech played
```

### Emotion Patterns
The system tracks your emotional journey:
- **Most common emotion**: Shows your predominant emotional state
- **Recent trend**: Indicates if you're feeling better, worse, or stable
- **Interaction history**: Maintains context across conversations

## 🛠️ Configuration

Edit `config.yaml` to customize:

```yaml
# Emotion detection sensitivity
emotion_detection:
  confidence_threshold: 0.6  # Lower = more sensitive

# Response style
response_generation:
  default_style: "supportive"  # supportive, reflective, solution_focused, therapeutic
  temperature: 0.7  # Higher = more creative responses

# Voice settings
text_to_speech:
  emotion_adaptive: true  # Adjust voice based on detected emotion
  speaking_rate: 1.0
```

## 🎨 Empathy Styles

### Supportive (Default)
- Validates emotions warmly
- Provides encouragement and comfort
- Example: *"I can hear the sadness in your voice, and I want you to know that it's okay to feel this way. You're not alone."*

### Reflective
- Mirrors back emotions and thoughts
- Helps process feelings
- Example: *"I notice a real heaviness in how you're speaking. You're expressing genuine sadness right now."*

### Solution-Focused
- Acknowledges emotions while guiding toward solutions
- Practical and forward-looking
- Example: *"I hear your frustration, and it sounds like something important needs to be addressed. What would need to change for you to feel better?"*

### Therapeutic
- Uses CBT-inspired techniques
- Helps reframe thoughts and feelings
- Example: *"Sadness can be difficult to sit with, but it's also a sign that something matters to you. What thoughts are accompanying this sadness?"*

## 🔧 Advanced Features

### Custom Voice Models
Add your own Piper TTS voices:
1. Download `.onnx` voice files from [Piper Voices](https://huggingface.co/rhasspy/piper-voices)
2. Place in `models/piper_voices/`
3. Update `config.yaml` voice mappings

### LLM Integration
For more sophisticated responses:
1. Install Ollama: https://ollama.ai
2. Run: `ollama pull llama3.1:8b`
3. The system will automatically use Ollama for enhanced responses

### Training Custom Emotion Models
```python
from src.emotion.detector import EmotionDetector

detector = EmotionDetector()
# Provide your own labeled audio files
detector.train_model(audio_files, labels)
```

## 🐛 Troubleshooting

### Audio Issues
```bash
# Test audio devices
python -c "import pyaudio; p=pyaudio.PyAudio(); print(f'Found {p.get_device_count()} audio devices')"

# On Linux, you may need:
sudo apt-get install portaudio19-dev python3-pyaudio

# On macOS:
brew install portaudio
```

### Speech Recognition Issues
- Ensure microphone permissions are granted
- Check microphone levels in system settings
- Try different Whisper models: `tiny`, `base`, `small`, `medium`, `large`

### TTS Issues
```bash
# Test system TTS
# Windows: Uses SAPI (built-in)
# macOS: Uses 'say' command
# Linux: Install espeak: sudo apt-get install espeak
```

### Memory Issues
- Reduce model sizes in `config.yaml`
- Use `tiny` Whisper model for lower memory usage
- Disable conversation saving if needed

## 📊 Understanding the Output

### Emotion Detection
- **Confidence**: 0.0-1.0 scale, higher means more certain
- **Supported Emotions**: happy, sad, angry, anxious, calm, neutral
- **Features Used**: MFCC, pitch, spectral features, prosodic analysis

### Response Quality
- **Response Confidence**: How confident the AI is in its response
- **Generation Time**: How long it took to generate the response
- **Empathy Style**: Which approach was used

## 🔒 Privacy & Security

- **Local Processing**: All audio and text processing happens on your device
- **No External APIs**: Optional Ollama runs locally
- **Conversation Storage**: Saved locally in `data/conversations.json`
- **Data Control**: Delete conversation history anytime

## 🤝 Contributing

This is an open-source project! Ways to contribute:
- Report bugs and suggest features
- Improve emotion detection accuracy
- Add new empathy patterns
- Create additional voice models
- Enhance the web interface

## 📚 Technical Details

### Architecture
```
Audio Input → Whisper STT → Emotion Detection → LLM Response → Piper TTS → Audio Output
     ↓              ↓              ↓               ↓            ↓
  Microphone    Speech Text    Emotion+Conf    Empathic Text  Speakers
```

### Models Used
- **Speech Recognition**: OpenAI Whisper (open-source)
- **Emotion Detection**: Custom ML model using Librosa features
- **Response Generation**: Ollama (local) or template-based
- **Text-to-Speech**: Piper TTS (neural, fast)

### Performance
- **Latency**: ~2-5 seconds end-to-end
- **Memory**: ~2-4GB RAM (depending on models)
- **CPU**: Works on modern laptops, GPU optional

## 🎉 What's Next?

This system demonstrates the power of combining multiple open-source AI tools to create something genuinely helpful. Future enhancements could include:

- Multi-language support
- Video emotion detection
- Integration with mental health resources
- Mobile app version
- Voice cloning for personalized responses

Enjoy exploring empathic AI! 🤖💙
