"""Speech recognition module using OpenAI Whisper."""

import whisper
import numpy as np
import torch
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import tempfile
import soundfile as sf
from dataclasses import dataclass

from ..utils.config import config
from ..utils.audio import AudioProcessor

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of speech transcription."""
    text: str
    language: str
    confidence: float
    segments: List[Dict[str, Any]]
    processing_time: float


class WhisperRecognizer:
    """Speech recognition using OpenAI Whisper."""
    
    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.audio_processor = AudioProcessor(sample_rate=16000)
        
        logger.info(f"Initializing Whisper with model '{model_name}' on device '{self.device}'")
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info(f"Whisper model '{self.model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(self, 
                  audio: np.ndarray, 
                  language: Optional[str] = None,
                  temperature: float = 0.0,
                  beam_size: int = 5,
                  best_of: int = 5,
                  patience: float = 1.0) -> TranscriptionResult:
        """Transcribe audio to text."""
        import time
        start_time = time.time()
        
        try:
            # Ensure audio is in the correct format
            audio = self.audio_processor.normalize_audio(audio)
            
            # Whisper expects audio to be float32
            audio = audio.astype(np.float32)
            
            # Transcribe using Whisper
            result = self.model.transcribe(
                audio,
                language=language,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                verbose=False
            )
            
            processing_time = time.time() - start_time
            
            # Calculate average confidence from segments
            confidence = 0.0
            if result.get('segments'):
                confidences = [seg.get('avg_logprob', 0.0) for seg in result['segments']]
                confidence = np.mean(confidences) if confidences else 0.0
                # Convert log probability to confidence (approximate)
                confidence = max(0.0, min(1.0, (confidence + 1.0) / 2.0))
            
            return TranscriptionResult(
                text=result['text'].strip(),
                language=result.get('language', 'unknown'),
                confidence=confidence,
                segments=result.get('segments', []),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptionResult(
                text="",
                language="unknown",
                confidence=0.0,
                segments=[],
                processing_time=time.time() - start_time
            )
    
    def transcribe_file(self, file_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe audio file."""
        try:
            audio, _ = self.audio_processor.load_audio(file_path)
            return self.transcribe(audio, **kwargs)
        except Exception as e:
            logger.error(f"Failed to transcribe file {file_path}: {e}")
            raise
    
    def is_speech_detected(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if speech is detected in audio."""
        if len(audio) == 0:
            return False
        
        # Simple energy-based detection
        rms = self.audio_processor.calculate_rms(audio)
        return rms > threshold
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(whisper.tokenizer.LANGUAGES.keys())


class FasterWhisperRecognizer:
    """Speech recognition using Faster-Whisper for better performance."""
    
    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.audio_processor = AudioProcessor(sample_rate=16000)
        
        try:
            from faster_whisper import WhisperModel
            self.WhisperModel = WhisperModel
            logger.info(f"Using Faster-Whisper with model '{model_name}' on device '{self.device}'")
            self._load_model()
        except ImportError:
            logger.warning("Faster-Whisper not available, falling back to standard Whisper")
            self.model = None
    
    def _load_model(self):
        """Load the Faster-Whisper model."""
        try:
            self.model = self.WhisperModel(
                self.model_name, 
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            logger.info(f"Faster-Whisper model '{self.model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper model: {e}")
            self.model = None
    
    def transcribe(self, 
                  audio: np.ndarray, 
                  language: Optional[str] = None,
                  temperature: float = 0.0,
                  beam_size: int = 5,
                  best_of: int = 5,
                  patience: float = 1.0) -> TranscriptionResult:
        """Transcribe audio to text using Faster-Whisper."""
        if self.model is None:
            raise RuntimeError("Faster-Whisper model not loaded")
        
        import time
        start_time = time.time()
        
        try:
            # Ensure audio is in the correct format
            audio = self.audio_processor.normalize_audio(audio)
            audio = audio.astype(np.float32)
            
            # Transcribe using Faster-Whisper
            segments, info = self.model.transcribe(
                audio,
                language=language,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience
            )
            
            # Collect segments
            segment_list = []
            full_text = ""
            confidences = []
            
            for segment in segments:
                segment_dict = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'avg_logprob': segment.avg_logprob
                }
                segment_list.append(segment_dict)
                full_text += segment.text
                confidences.append(segment.avg_logprob)
            
            processing_time = time.time() - start_time
            
            # Calculate average confidence
            confidence = 0.0
            if confidences:
                confidence = np.mean(confidences)
                # Convert log probability to confidence (approximate)
                confidence = max(0.0, min(1.0, (confidence + 1.0) / 2.0))
            
            return TranscriptionResult(
                text=full_text.strip(),
                language=info.language,
                confidence=confidence,
                segments=segment_list,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Faster-Whisper transcription failed: {e}")
            return TranscriptionResult(
                text="",
                language="unknown",
                confidence=0.0,
                segments=[],
                processing_time=time.time() - start_time
            )

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # Faster-Whisper supports the same languages as Whisper
        try:
            import whisper
            return list(whisper.tokenizer.LANGUAGES.keys())
        except ImportError:
            # Fallback list of common languages
            return ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']

    def transcribe_file(self, file_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe audio file."""
        try:
            from ..utils.audio import AudioProcessor
            audio_processor = AudioProcessor(sample_rate=16000)
            audio, _ = audio_processor.load_audio(file_path)
            return self.transcribe(audio, **kwargs)
        except Exception as e:
            logger.error(f"Failed to transcribe file {file_path}: {e}")
            raise

    def is_speech_detected(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if speech is detected in audio."""
        if len(audio) == 0:
            return False

        # Simple energy-based detection
        from ..utils.audio import AudioProcessor
        audio_processor = AudioProcessor()
        rms = audio_processor.calculate_rms(audio)
        return rms > threshold


class SpeechRecognitionPipeline:
    """Main speech recognition pipeline."""
    
    def __init__(self):
        self.config = config.speech_recognition
        self.recognizer = None
        self._initialize_recognizer()
    
    def _initialize_recognizer(self):
        """Initialize the appropriate recognizer."""
        try:
            # Try Faster-Whisper first for better performance
            self.recognizer = FasterWhisperRecognizer(
                model_name=self.config.model,
                device=None  # Auto-detect
            )
            if self.recognizer.model is None:
                raise RuntimeError("Faster-Whisper not available")
            logger.info("Using Faster-Whisper for speech recognition")
        except:
            # Fall back to standard Whisper
            self.recognizer = WhisperRecognizer(
                model_name=self.config.model,
                device=None  # Auto-detect
            )
            logger.info("Using standard Whisper for speech recognition")
    
    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe audio using configured settings."""
        return self.recognizer.transcribe(
            audio,
            language=self.config.language if self.config.language != "auto" else None,
            temperature=self.config.temperature,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            patience=self.config.patience
        )
    
    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """Transcribe audio file."""
        return self.recognizer.transcribe_file(file_path)
    
    def is_speech_detected(self, audio: np.ndarray) -> bool:
        """Check if speech is detected in audio."""
        return self.recognizer.is_speech_detected(audio)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.recognizer.get_supported_languages()
