"""Configuration management for Empathic Voice Companion."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    silence_threshold: float = 0.01
    silence_duration: float = 2.0


@dataclass
class SpeechRecognitionConfig:
    model: str = "base"
    language: str = "en"
    temperature: float = 0.0
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0


@dataclass
class EmotionDetectionConfig:
    enabled: bool = True
    confidence_threshold: float = 0.6
    supported_emotions: list = None
    mfcc_coefficients: int = 13
    chroma_features: int = 12
    spectral_features: bool = True
    prosodic_features: bool = True
    window_size: int = 2048
    hop_length: int = 512

    def __post_init__(self):
        if self.supported_emotions is None:
            self.supported_emotions = ["happy", "sad", "angry", "anxious", "calm", "neutral"]


@dataclass
class ResponseGenerationConfig:
    llm_provider: str = "ollama"
    model_name: str = "llama3.1:8b"
    max_tokens: int = 150
    temperature: float = 0.7
    default_style: str = "supportive"
    empathy_styles: Dict[str, str] = None

    def __post_init__(self):
        if self.empathy_styles is None:
            self.empathy_styles = {
                "supportive": "Provide warm, encouraging responses that validate emotions",
                "reflective": "Mirror back emotions and help process feelings",
                "solution_focused": "Acknowledge emotions while gently guiding toward solutions",
                "therapeutic": "Use CBT-inspired techniques for emotional support"
            }


@dataclass
class TextToSpeechConfig:
    provider: str = "piper"
    voice_model: str = "en_US-lessac-medium"
    speaking_rate: float = 1.0
    pitch_adjustment: float = 0.0
    volume: float = 0.8
    emotion_adaptive: bool = True
    voice_mapping: Dict[str, str] = None

    def __post_init__(self):
        if self.voice_mapping is None:
            self.voice_mapping = {
                "happy": "en_US-amy-medium",
                "sad": "en_US-lessac-low",
                "angry": "en_US-ryan-high",
                "anxious": "en_US-lessac-medium",
                "calm": "en_US-lessac-low",
                "neutral": "en_US-lessac-medium"
            }


@dataclass
class MemoryConfig:
    enabled: bool = True
    max_history_length: int = 50
    emotion_history_window: int = 10
    context_window: int = 5
    save_conversations: bool = True
    conversation_file: str = "data/conversations.json"


class Config:
    """Main configuration class for the Empathic Voice Companion."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config_data = {}
        self.load_config()
        
        # Initialize configuration sections
        self.audio = self._create_audio_config()
        self.speech_recognition = self._create_speech_recognition_config()
        self.emotion_detection = self._create_emotion_detection_config()
        self.response_generation = self._create_response_generation_config()
        self.text_to_speech = self._create_text_to_speech_config()
        self.memory = self._create_memory_config()
    
    def load_config(self):
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f) or {}
        else:
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            self._config_data = {}
    
    def save_config(self):
        """Save current configuration to YAML file."""
        config_dict = {
            'audio': self.audio.__dict__,
            'speech_recognition': self.speech_recognition.__dict__,
            'emotion_detection': self.emotion_detection.__dict__,
            'response_generation': self.response_generation.__dict__,
            'text_to_speech': self.text_to_speech.__dict__,
            'memory': self.memory.__dict__
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _create_audio_config(self) -> AudioConfig:
        audio_data = self._config_data.get('audio', {})
        return AudioConfig(**audio_data)
    
    def _create_speech_recognition_config(self) -> SpeechRecognitionConfig:
        sr_data = self._config_data.get('speech_recognition', {})
        return SpeechRecognitionConfig(**sr_data)
    
    def _create_emotion_detection_config(self) -> EmotionDetectionConfig:
        ed_data = self._config_data.get('emotion_detection', {})
        feature_data = ed_data.get('feature_extraction', {})
        
        # Merge feature extraction settings
        merged_data = {**ed_data, **feature_data}
        merged_data.pop('feature_extraction', None)
        
        return EmotionDetectionConfig(**merged_data)
    
    def _create_response_generation_config(self) -> ResponseGenerationConfig:
        rg_data = self._config_data.get('response_generation', {})
        return ResponseGenerationConfig(**rg_data)
    
    def _create_text_to_speech_config(self) -> TextToSpeechConfig:
        tts_data = self._config_data.get('text_to_speech', {})
        return TextToSpeechConfig(**tts_data)
    
    def _create_memory_config(self) -> MemoryConfig:
        memory_data = self._config_data.get('memory', {})
        return MemoryConfig(**memory_data)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key path (e.g., 'web.host')."""
        keys = key.split('.')
        value = self._config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_model_path(self, model_type: str) -> Path:
        """Get the path for a specific model type."""
        models_config = self._config_data.get('models', {})
        base_path = Path(models_config.get(model_type, f"models/{model_type}"))
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path


# Global configuration instance
config = Config()
