"""Text-to-speech synthesis for empathic responses."""

import numpy as np
import logging
import tempfile
import subprocess
import os
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import time

from ..utils.config import config
from ..utils.audio import AudioProcessor

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Result of text-to-speech synthesis."""
    audio: np.ndarray
    sample_rate: int
    synthesis_time: float
    voice_used: str
    success: bool


class PiperTTSProvider:
    """Piper TTS provider for high-quality speech synthesis."""
    
    def __init__(self):
        self.config = config.text_to_speech
        self.audio_processor = AudioProcessor(sample_rate=config.audio.sample_rate)
        self.voices_path = config.get_model_path("piper_voices")
        self.available_voices = self._discover_voices()
    
    def _discover_voices(self) -> Dict[str, str]:
        """Discover available Piper voices."""
        voices = {}
        
        # Check for downloaded voice models
        if self.voices_path.exists():
            for voice_file in self.voices_path.glob("*.onnx"):
                voice_name = voice_file.stem
                voices[voice_name] = str(voice_file)
        
        # Add some default voice mappings
        default_voices = {
            "en_US-lessac-medium": "en_US-lessac-medium.onnx",
            "en_US-amy-medium": "en_US-amy-medium.onnx",
            "en_US-ryan-high": "en_US-ryan-high.onnx"
        }
        
        for voice_name, filename in default_voices.items():
            if voice_name not in voices:
                voices[voice_name] = filename
        
        return voices
    
    def synthesize(self, text: str, voice: Optional[str] = None) -> SynthesisResult:
        """Synthesize speech from text using Piper."""
        start_time = time.time()

        try:
            # Choose voice
            voice_name = voice or self.config.voice_model

            # If using system provider or voice is "default", skip Piper
            if self.config.provider == "system" or voice_name == "default":
                return self._synthesize_with_system_tts(text, start_time)

            # Try to use piper-tts if available
            audio = self._synthesize_with_piper(text, voice_name)

            if audio is not None:
                synthesis_time = time.time() - start_time
                return SynthesisResult(
                    audio=audio,
                    sample_rate=self.audio_processor.sample_rate,
                    synthesis_time=synthesis_time,
                    voice_used=voice_name,
                    success=True
                )
            else:
                # Fall back to system TTS
                return self._synthesize_with_system_tts(text, start_time)

        except Exception as e:
            logger.error(f"Piper TTS synthesis failed: {e}")
            return self._synthesize_with_system_tts(text, start_time)
    
    def _synthesize_with_piper(self, text: str, voice: str) -> Optional[np.ndarray]:
        """Synthesize using Piper TTS."""
        try:
            # Check if piper is available
            result = subprocess.run(["piper", "--help"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                logger.warning("Piper TTS not found in PATH")
                return None
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as text_file:
                text_file.write(text)
                text_file_path = text_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
                audio_file_path = audio_file.name
            
            try:
                # Run Piper TTS
                cmd = [
                    "piper",
                    "--model", voice,
                    "--output_file", audio_file_path
                ]
                
                with open(text_file_path, 'r') as input_file:
                    result = subprocess.run(
                        cmd,
                        stdin=input_file,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                
                if result.returncode == 0 and os.path.exists(audio_file_path):
                    # Load generated audio
                    audio, _ = self.audio_processor.load_audio(audio_file_path)
                    return audio
                else:
                    logger.warning(f"Piper TTS failed: {result.stderr}")
                    return None
                    
            finally:
                # Clean up temporary files
                try:
                    os.unlink(text_file_path)
                    os.unlink(audio_file_path)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            logger.warning("Piper TTS timed out")
            return None
        except FileNotFoundError:
            logger.warning("Piper TTS not installed")
            return None
        except Exception as e:
            logger.warning(f"Piper TTS error: {e}")
            return None
    
    def _synthesize_with_system_tts(self, text: str, start_time: float) -> SynthesisResult:
        """Fallback to system TTS."""
        try:
            # Try different system TTS options
            audio = None
            
            # Windows SAPI
            if os.name == 'nt':
                audio = self._synthesize_with_sapi(text)
            
            # macOS say command
            elif os.name == 'posix' and os.uname().sysname == 'Darwin':
                audio = self._synthesize_with_say(text)
            
            # Linux espeak
            elif os.name == 'posix':
                audio = self._synthesize_with_espeak(text)
            
            if audio is not None:
                synthesis_time = time.time() - start_time
                return SynthesisResult(
                    audio=audio,
                    sample_rate=self.audio_processor.sample_rate,
                    synthesis_time=synthesis_time,
                    voice_used="system",
                    success=True
                )
            
        except Exception as e:
            logger.error(f"System TTS failed: {e}")
        
        # Return empty result if all fails
        synthesis_time = time.time() - start_time
        return SynthesisResult(
            audio=np.array([]),
            sample_rate=self.audio_processor.sample_rate,
            synthesis_time=synthesis_time,
            voice_used="none",
            success=False
        )
    
    def _synthesize_with_sapi(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using Windows SAPI."""
        try:
            import win32com.client

            # Create SAPI voice object
            voice = win32com.client.Dispatch("SAPI.SpVoice")

            # Create file stream
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name

            file_stream = win32com.client.Dispatch("SAPI.SpFileStream")
            file_stream.Open(temp_path, 3)  # Write mode
            voice.AudioOutputStream = file_stream

            # Speak to file
            voice.Speak(text)
            file_stream.Close()

            # Load audio
            audio, _ = self.audio_processor.load_audio(temp_path)

            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass

            return audio

        except ImportError:
            logger.warning("pywin32 not available for SAPI TTS")
            return None
        except Exception as e:
            logger.warning(f"SAPI TTS error: {e}")
            return None
    
    def _synthesize_with_say(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using macOS say command."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Run say command
            result = subprocess.run([
                "say", "-o", temp_path, text
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(temp_path):
                audio, _ = self.audio_processor.load_audio(temp_path)
                os.unlink(temp_path)
                return audio
            
            return None
            
        except Exception as e:
            logger.warning(f"macOS say command error: {e}")
            return None
    
    def _synthesize_with_espeak(self, text: str) -> Optional[np.ndarray]:
        """Synthesize using espeak."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Run espeak
            result = subprocess.run([
                "espeak", "-w", temp_path, text
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(temp_path):
                audio, _ = self.audio_processor.load_audio(temp_path)
                os.unlink(temp_path)
                return audio
            
            return None
            
        except Exception as e:
            logger.warning(f"espeak error: {e}")
            return None
    
    def get_available_voices(self) -> Dict[str, str]:
        """Get available voices."""
        return self.available_voices.copy()
    
    def is_available(self) -> bool:
        """Check if TTS is available."""
        # Check for Piper
        try:
            result = subprocess.run(["piper", "--help"], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                return True
        except:
            pass
        
        # Check for system TTS
        if os.name == 'nt':
            try:
                import win32com.client
                return True
            except ImportError:
                pass
        
        elif os.name == 'posix':
            try:
                # Check for say (macOS) or espeak (Linux)
                for cmd in ["say", "espeak"]:
                    result = subprocess.run([cmd, "--help"], 
                                          capture_output=True, timeout=5)
                    if result.returncode == 0:
                        return True
            except:
                pass
        
        return False


class TextToSpeechPipeline:
    """Main text-to-speech pipeline."""
    
    def __init__(self):
        self.config = config.text_to_speech
        self.provider = PiperTTSProvider()
    
    def synthesize(self, text: str, emotion: Optional[str] = None) -> SynthesisResult:
        """Synthesize speech with emotion-appropriate voice."""
        # Choose voice based on emotion
        voice = self._choose_voice_for_emotion(emotion)
        
        # Adjust text for better synthesis
        text = self._prepare_text_for_synthesis(text)
        
        # Synthesize
        result = self.provider.synthesize(text, voice)
        
        # Apply post-processing if needed
        if result.success and self.config.emotion_adaptive:
            result.audio = self._apply_emotion_processing(result.audio, emotion)
        
        return result
    
    def _choose_voice_for_emotion(self, emotion: Optional[str]) -> str:
        """Choose appropriate voice for emotion."""
        if not emotion or not self.config.emotion_adaptive:
            return self.config.voice_model
        
        # Get voice mapping for emotion
        voice = self.config.voice_mapping.get(emotion, self.config.voice_model)
        
        # Check if voice is available
        available_voices = self.provider.get_available_voices()
        if voice not in available_voices:
            return self.config.voice_model
        
        return voice
    
    def _prepare_text_for_synthesis(self, text: str) -> str:
        """Prepare text for better synthesis."""
        # Remove or replace problematic characters
        text = text.replace('"', '')
        text = text.replace("'", '')
        
        # Add pauses for better pacing
        text = text.replace('. ', '. <break time="0.5s"/> ')
        text = text.replace('? ', '? <break time="0.3s"/> ')
        text = text.replace('! ', '! <break time="0.3s"/> ')
        
        # Limit length
        if len(text) > 500:
            sentences = text.split('. ')
            text = '. '.join(sentences[:3])
            if not text.endswith('.'):
                text += '.'
        
        return text
    
    def _apply_emotion_processing(self, audio: np.ndarray, emotion: Optional[str]) -> np.ndarray:
        """Apply emotion-specific audio processing."""
        if not emotion or len(audio) == 0:
            return audio
        
        try:
            # Simple emotion-based processing
            if emotion == 'happy':
                # Slightly increase pitch and tempo
                audio = self._adjust_pitch(audio, factor=1.1)
            elif emotion == 'sad':
                # Slightly decrease pitch and tempo
                audio = self._adjust_pitch(audio, factor=0.9)
            elif emotion == 'angry':
                # Increase volume and add slight distortion
                audio = audio * 1.2
                audio = np.clip(audio, -1.0, 1.0)
            elif emotion == 'calm':
                # Smooth and soften
                audio = self._smooth_audio(audio)
            
            return audio
            
        except Exception as e:
            logger.warning(f"Emotion processing failed: {e}")
            return audio
    
    def _adjust_pitch(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Simple pitch adjustment using resampling."""
        try:
            import librosa
            # Use librosa for pitch shifting
            return librosa.effects.pitch_shift(audio, sr=self.provider.audio_processor.sample_rate, n_steps=factor)
        except ImportError:
            logger.warning("Librosa not available for pitch shifting, using simple resampling")
        except Exception as e:
            logger.warning(f"Pitch shifting failed: {e}")

        # Fallback: simple time stretching
        try:
            if factor > 1.0:
                # Higher pitch: compress time
                indices = np.arange(0, len(audio), 1/factor).astype(int)
                indices = indices[indices < len(audio)]
                return audio[indices]
            else:
                # Lower pitch: stretch time
                try:
                    from scipy import interpolate
                    x_old = np.arange(len(audio))
                    x_new = np.arange(0, len(audio), factor)
                    f = interpolate.interp1d(x_old, audio, kind='linear', fill_value='extrapolate')
                    return f(x_new)
                except ImportError:
                    logger.warning("Scipy not available for interpolation")
                    return audio
        except Exception as e:
            logger.warning(f"Fallback pitch adjustment failed: {e}")
            return audio
    
    def _smooth_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply smoothing to audio."""
        try:
            from scipy import ndimage
            return ndimage.gaussian_filter1d(audio, sigma=1.0)
        except:
            # Simple moving average
            window_size = 5
            kernel = np.ones(window_size) / window_size
            return np.convolve(audio, kernel, mode='same')
    
    def is_available(self) -> bool:
        """Check if TTS is available."""
        return self.provider.is_available()
    
    def get_available_voices(self) -> Dict[str, str]:
        """Get available voices."""
        return self.provider.get_available_voices()
