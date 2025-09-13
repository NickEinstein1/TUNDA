"""Audio feature extraction for emotion detection."""

import numpy as np
import librosa
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..utils.config import config
from ..utils.audio import AudioProcessor

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Container for extracted audio features."""
    mfcc: np.ndarray
    chroma: np.ndarray
    spectral_centroid: np.ndarray
    spectral_bandwidth: np.ndarray
    spectral_rolloff: np.ndarray
    zero_crossing_rate: np.ndarray
    rms_energy: np.ndarray
    tempo: float
    pitch_mean: float
    pitch_std: float
    formants: List[float]
    jitter: float
    shimmer: float
    hnr: float  # Harmonics-to-Noise Ratio


class AudioFeatureExtractor:
    """Extract audio features for emotion detection."""
    
    def __init__(self):
        self.config = config.emotion_detection
        self.audio_processor = AudioProcessor(sample_rate=config.audio.sample_rate)
        self.sample_rate = config.audio.sample_rate
    
    def extract_features(self, audio: np.ndarray) -> AudioFeatures:
        """Extract comprehensive audio features."""
        try:
            # Ensure audio is normalized and properly formatted
            audio = self.audio_processor.normalize_audio(audio)
            
            if len(audio) == 0:
                return self._get_empty_features()
            
            # Extract different types of features
            mfcc = self._extract_mfcc(audio)
            chroma = self._extract_chroma(audio)
            spectral_features = self._extract_spectral_features(audio)
            prosodic_features = self._extract_prosodic_features(audio)
            
            return AudioFeatures(
                mfcc=mfcc,
                chroma=chroma,
                spectral_centroid=spectral_features['centroid'],
                spectral_bandwidth=spectral_features['bandwidth'],
                spectral_rolloff=spectral_features['rolloff'],
                zero_crossing_rate=spectral_features['zcr'],
                rms_energy=spectral_features['rms'],
                tempo=prosodic_features['tempo'],
                pitch_mean=prosodic_features['pitch_mean'],
                pitch_std=prosodic_features['pitch_std'],
                formants=prosodic_features['formants'],
                jitter=prosodic_features['jitter'],
                shimmer=prosodic_features['shimmer'],
                hnr=prosodic_features['hnr']
            )
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._get_empty_features()
    
    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features."""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.config.mfcc_coefficients,
                n_fft=self.config.window_size,
                hop_length=self.config.hop_length
            )
            return np.mean(mfcc, axis=1)  # Take mean across time
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            return np.zeros(self.config.mfcc_coefficients)
    
    def _extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features."""
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.config.window_size,
                hop_length=self.config.hop_length
            )
            return np.mean(chroma, axis=1)  # Take mean across time
        except Exception as e:
            logger.warning(f"Chroma extraction failed: {e}")
            return np.zeros(self.config.chroma_features)
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spectral features."""
        features = {}
        
        try:
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate,
                hop_length=self.config.hop_length
            )
            features['centroid'] = np.mean(centroid)
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sample_rate,
                hop_length=self.config.hop_length
            )
            features['bandwidth'] = np.mean(bandwidth)
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate,
                hop_length=self.config.hop_length
            )
            features['rolloff'] = np.mean(rolloff)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                audio, hop_length=self.config.hop_length
            )
            features['zcr'] = np.mean(zcr)
            
            # RMS energy
            rms = librosa.feature.rms(
                y=audio, hop_length=self.config.hop_length
            )
            features['rms'] = np.mean(rms)
            
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")
            features = {
                'centroid': 0.0,
                'bandwidth': 0.0,
                'rolloff': 0.0,
                'zcr': 0.0,
                'rms': 0.0
            }
        
        return features
    
    def _extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract prosodic features (pitch, tempo, etc.)."""
        features = {
            'tempo': 0.0,
            'pitch_mean': 0.0,
            'pitch_std': 0.0,
            'formants': [0.0, 0.0, 0.0],
            'jitter': 0.0,
            'shimmer': 0.0,
            'hnr': 0.0
        }
        
        try:
            # Tempo estimation
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            features['tempo'] = float(tempo)
            
            # Pitch extraction using piptrack
            pitches, magnitudes = librosa.piptrack(
                y=audio, sr=self.sample_rate,
                hop_length=self.config.hop_length
            )
            
            # Extract pitch values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                
                # Calculate jitter (pitch variation)
                if len(pitch_values) > 1:
                    pitch_diffs = np.abs(np.diff(pitch_values))
                    features['jitter'] = np.mean(pitch_diffs) / features['pitch_mean']
            
            # Estimate formants (simplified)
            features['formants'] = self._estimate_formants(audio)
            
            # Shimmer (amplitude variation)
            features['shimmer'] = self._calculate_shimmer(audio)
            
            # Harmonics-to-Noise Ratio
            features['hnr'] = self._calculate_hnr(audio)
            
        except Exception as e:
            logger.warning(f"Prosodic feature extraction failed: {e}")
        
        return features
    
    def _estimate_formants(self, audio: np.ndarray, n_formants: int = 3) -> List[float]:
        """Estimate formant frequencies (simplified method)."""
        try:
            # Use spectral analysis to estimate formants
            # from scipy.signal import lfilter
            
            # Pre-emphasis
            audio_preemph = self.audio_processor.apply_pre_emphasis(audio)
            
            # Window the signal
            windowed = audio_preemph * np.hanning(len(audio_preemph))
            
            # Autocorrelation method for LPC
            # This is a simplified formant estimation
            fft = np.fft.fft(windowed, n=2048)
            magnitude = np.abs(fft)
            
            # Find peaks in the spectrum (rough formant estimation)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(magnitude[:1024], height=np.max(magnitude) * 0.1)
            
            # Convert to frequencies
            formant_freqs = peaks * self.sample_rate / 2048
            
            # Return first n_formants
            formants = formant_freqs[:n_formants].tolist()
            while len(formants) < n_formants:
                formants.append(0.0)
            
            return formants
            
        except Exception as e:
            logger.warning(f"Formant estimation failed: {e}")
            return [0.0] * n_formants
    
    def _calculate_shimmer(self, audio: np.ndarray) -> float:
        """Calculate shimmer (amplitude variation)."""
        try:
            # Frame-based amplitude calculation
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.010 * self.sample_rate)    # 10ms hop
            
            amplitudes = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                amplitude = np.max(np.abs(frame))
                if amplitude > 0:
                    amplitudes.append(amplitude)
            
            if len(amplitudes) > 1:
                amp_diffs = np.abs(np.diff(amplitudes))
                mean_amplitude = np.mean(amplitudes)
                if mean_amplitude > 0:
                    return np.mean(amp_diffs) / mean_amplitude
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Shimmer calculation failed: {e}")
            return 0.0
    
    def _calculate_hnr(self, audio: np.ndarray) -> float:
        """Calculate Harmonics-to-Noise Ratio."""
        try:
            # Autocorrelation-based HNR estimation
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            if len(autocorr) > 1:
                # Find the maximum autocorrelation (excluding zero lag)
                max_autocorr = np.max(autocorr[1:])
                noise_level = autocorr[0] - max_autocorr
                
                if noise_level > 0:
                    hnr = 10 * np.log10(max_autocorr / noise_level)
                    return max(0.0, min(40.0, hnr))  # Clamp to reasonable range
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"HNR calculation failed: {e}")
            return 0.0
    
    def _get_empty_features(self) -> AudioFeatures:
        """Return empty features for error cases."""
        return AudioFeatures(
            mfcc=np.zeros(self.config.mfcc_coefficients),
            chroma=np.zeros(self.config.chroma_features),
            spectral_centroid=0.0,
            spectral_bandwidth=0.0,
            spectral_rolloff=0.0,
            zero_crossing_rate=0.0,
            rms_energy=0.0,
            tempo=0.0,
            pitch_mean=0.0,
            pitch_std=0.0,
            formants=[0.0, 0.0, 0.0],
            jitter=0.0,
            shimmer=0.0,
            hnr=0.0
        )
    
    def features_to_vector(self, features: AudioFeatures) -> np.ndarray:
        """Convert AudioFeatures to a feature vector for ML models."""
        vector = np.concatenate([
            features.mfcc,
            features.chroma,
            [features.spectral_centroid],
            [features.spectral_bandwidth],
            [features.spectral_rolloff],
            [features.zero_crossing_rate],
            [features.rms_energy],
            [features.tempo],
            [features.pitch_mean],
            [features.pitch_std],
            features.formants,
            [features.jitter],
            [features.shimmer],
            [features.hnr]
        ])
        
        # Handle any NaN or infinite values
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        return vector
