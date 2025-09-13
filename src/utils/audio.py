"""Audio processing utilities for the Empathic Voice Companion."""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio processing operations."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate."""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise
    
    def save_audio(self, audio: np.ndarray, file_path: str, sample_rate: Optional[int] = None):
        """Save audio data to file."""
        sr = sample_rate or self.sample_rate
        try:
            sf.write(file_path, audio, sr)
            logger.info(f"Audio saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving audio to {file_path}: {e}")
            raise
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        if len(audio) == 0:
            return audio
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def remove_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Remove silence from beginning and end of audio."""
        if len(audio) == 0:
            return audio
        
        # Find non-silent regions
        non_silent = np.abs(audio) > threshold
        
        if not np.any(non_silent):
            return audio
        
        # Find first and last non-silent samples
        first_sound = np.argmax(non_silent)
        last_sound = len(audio) - np.argmax(non_silent[::-1]) - 1
        
        return audio[first_sound:last_sound + 1]
    
    def apply_pre_emphasis(self, audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter to audio."""
        if len(audio) <= 1:
            return audio
        
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
    
    def split_audio_chunks(self, audio: np.ndarray, chunk_duration: float = 1.0) -> List[np.ndarray]:
        """Split audio into chunks of specified duration."""
        chunk_samples = int(chunk_duration * self.sample_rate)
        chunks = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) > 0:
                chunks.append(chunk)
        
        return chunks
    
    def detect_voice_activity(self, audio: np.ndarray, 
                            frame_length: int = 2048, 
                            hop_length: int = 512,
                            threshold: float = 0.01) -> np.ndarray:
        """Detect voice activity in audio using energy-based method."""
        # Calculate frame-wise energy
        frames = librosa.util.frame(audio, frame_length=frame_length, 
                                  hop_length=hop_length, axis=0)
        energy = np.sum(frames ** 2, axis=0)
        
        # Normalize energy
        if np.max(energy) > 0:
            energy = energy / np.max(energy)
        
        # Apply threshold
        voice_activity = energy > threshold
        
        return voice_activity
    
    def apply_noise_reduction(self, audio: np.ndarray, 
                            noise_duration: float = 0.5) -> np.ndarray:
        """Simple noise reduction using spectral subtraction."""
        try:
            # Estimate noise from the first portion of audio
            noise_samples = int(noise_duration * self.sample_rate)
            if len(audio) <= noise_samples:
                return audio
            
            noise_segment = audio[:noise_samples]
            
            # Compute STFT
            stft = librosa.stft(audio)
            noise_stft = librosa.stft(noise_segment)
            
            # Estimate noise power spectrum
            noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
            
            # Apply spectral subtraction
            signal_power = np.abs(stft) ** 2
            clean_power = signal_power - 2 * noise_power
            clean_power = np.maximum(clean_power, 0.1 * signal_power)
            
            # Reconstruct audio
            clean_magnitude = np.sqrt(clean_power)
            clean_stft = clean_magnitude * np.exp(1j * np.angle(stft))
            clean_audio = librosa.istft(clean_stft)
            
            return clean_audio
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}. Returning original audio.")
            return audio
    
    def get_audio_duration(self, audio: np.ndarray) -> float:
        """Get duration of audio in seconds."""
        return len(audio) / self.sample_rate
    
    def resample_audio(self, audio: np.ndarray, 
                      original_sr: int, 
                      target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if original_sr == target_sr:
            return audio
        
        return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    
    def convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo audio to mono."""
        if audio.ndim == 1:
            return audio
        elif audio.ndim == 2:
            return np.mean(audio, axis=1)
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")
    
    def add_padding(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Add zero padding to audio to reach target length."""
        if len(audio) >= target_length:
            return audio[:target_length]
        
        padding = target_length - len(audio)
        return np.pad(audio, (0, padding), mode='constant', constant_values=0)
    
    def calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) of audio."""
        if len(audio) == 0:
            return 0.0
        return np.sqrt(np.mean(audio ** 2))
    
    def calculate_zero_crossing_rate(self, audio: np.ndarray) -> float:
        """Calculate zero crossing rate of audio."""
        if len(audio) <= 1:
            return 0.0
        
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
        return zero_crossings / len(audio)


class AudioBuffer:
    """Circular buffer for real-time audio processing."""
    
    def __init__(self, max_duration: float, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.is_full = False
    
    def write(self, audio: np.ndarray):
        """Write audio data to buffer."""
        audio = audio.astype(np.float32)
        
        for sample in audio:
            self.buffer[self.write_pos] = sample
            self.write_pos = (self.write_pos + 1) % self.max_samples
            
            if self.write_pos == 0:
                self.is_full = True
    
    def read(self, duration: float) -> np.ndarray:
        """Read audio data from buffer."""
        samples_to_read = int(duration * self.sample_rate)
        samples_to_read = min(samples_to_read, self.get_available_samples())
        
        if samples_to_read == 0:
            return np.array([], dtype=np.float32)
        
        if self.is_full:
            # Read from current position backwards
            start_pos = (self.write_pos - samples_to_read) % self.max_samples
            if start_pos + samples_to_read <= self.max_samples:
                return self.buffer[start_pos:start_pos + samples_to_read].copy()
            else:
                # Wrap around
                part1 = self.buffer[start_pos:].copy()
                part2 = self.buffer[:samples_to_read - len(part1)].copy()
                return np.concatenate([part1, part2])
        else:
            # Buffer not full yet
            start_pos = max(0, self.write_pos - samples_to_read)
            return self.buffer[start_pos:self.write_pos].copy()
    
    def get_available_samples(self) -> int:
        """Get number of available samples in buffer."""
        if self.is_full:
            return self.max_samples
        else:
            return self.write_pos
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.fill(0)
        self.write_pos = 0
        self.is_full = False
