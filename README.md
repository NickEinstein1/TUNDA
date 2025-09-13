# Empathic Voice Companion

An AI-powered voice assistant that detects emotional states from speech and responds with appropriate empathy using open-source tools.

## Features

- **Real-time Speech Recognition** - Powered by OpenAI Whisper
- **Emotion Detection** - Analyzes voice tone, pitch, and prosodic features
- **Empathic Response Generation** - Context-aware responses using local LLM
- **Adaptive Text-to-Speech** - Voice output that matches emotional context
- **Conversation Memory** - Tracks emotional context across interactions
- **Privacy-First** - All processing happens locally, no data sent to external services

## Architecture

```
Audio Input → Speech Recognition → Emotion Detection → Response Generation → Text-to-Speech → Audio Output
     ↓              ↓                    ↓                     ↓                  ↓
  Microphone    Whisper STT         Librosa +            Local LLM          Piper TTS
                                   ML Classifier        (Ollama/HF)
```

## Supported Emotions

- **Happy** - Joyful, excited, positive
- **Sad** - Melancholic, disappointed, down
- **Angry** - Frustrated, irritated, upset
- **Anxious** - Worried, stressed, nervous
- **Calm** - Peaceful, relaxed, content
- **Neutral** - Balanced, matter-of-fact

## Installation

### Prerequisites

- Python 3.8+ (3.9-3.11 recommended)
- FFmpeg (for audio processing)
- At least 4GB RAM (for local LLM)

### Quick Setup (Recommended)

1. Clone or download the project:
```bash
cd empathic-voice-companion
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Run the safe installer:
```bash
python install.py
```

This installer handles dependency compatibility issues automatically.

### Alternative Setup

If you prefer manual installation:
```bash
# Install minimal dependencies
pip install -r requirements-minimal.txt

# Then run setup for models
python setup_models.py
```

## Usage

### Basic Usage
```bash
python main.py
```

### Web Interface
```bash
python app.py
# Open http://localhost:8000 in your browser
```

### API Mode
```bash
python api_server.py
# API available at http://localhost:8001
```

## Configuration

Edit `config.yaml` to customize:
- Emotion detection sensitivity
- Response personality styles
- Voice models and settings
- Audio input/output devices

## Development

### Project Structure
```
empathic-voice-companion/
├── src/
│   ├── speech/
│   │   ├── recognition.py      # Whisper STT integration
│   │   └── synthesis.py        # Piper TTS integration
│   ├── emotion/
│   │   ├── detector.py         # Emotion detection engine
│   │   └── features.py         # Audio feature extraction
│   ├── response/
│   │   ├── generator.py        # LLM response generation
│   │   └── empathy.py          # Empathic response patterns
│   ├── memory/
│   │   └── conversation.py     # Conversation history
│   └── utils/
│       ├── audio.py            # Audio processing utilities
│       └── config.py           # Configuration management
├── models/                     # Downloaded AI models
├── data/                       # Training data and samples
├── tests/                      # Unit tests
├── web/                        # Web interface files
├── main.py                     # Main CLI application
├── app.py                      # Web application
├── api_server.py              # REST API server
├── requirements.txt           # Python dependencies
├── config.yaml               # Configuration file
└── setup_models.py           # Model download script
```

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Acknowledgments

- OpenAI Whisper for speech recognition
- Librosa for audio analysis
- Piper TTS for speech synthesis
- Hugging Face for ML models
