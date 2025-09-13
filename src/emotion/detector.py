"""Emotion detection from audio features."""

import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from ..utils.config import config
from .features import AudioFeatureExtractor, AudioFeatures

logger = logging.getLogger(__name__)


@dataclass
class EmotionPrediction:
    """Result of emotion prediction."""
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    features_used: int


class EmotionDetector:
    """Emotion detection from audio features."""
    
    def __init__(self):
        self.config = config.emotion_detection
        self.feature_extractor = AudioFeatureExtractor()
        self.model = None
        self.scaler = None
        self.emotion_labels = self.config.supported_emotions
        self.model_path = config.get_model_path("emotion_model")
        
        # Try to load existing model
        self._load_model()
        
        # If no model exists, create and train a basic one
        if self.model is None:
            self._create_default_model()
    
    def _load_model(self):
        """Load trained emotion detection model."""
        model_file = self.model_path / "emotion_classifier.pkl"
        scaler_file = self.model_path / "feature_scaler.pkl"
        
        try:
            if model_file.exists() and scaler_file.exists():
                self.model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                logger.info("Loaded existing emotion detection model")
            else:
                logger.info("No existing emotion model found")
        except Exception as e:
            logger.warning(f"Failed to load emotion model: {e}")
            self.model = None
            self.scaler = None
    
    def _save_model(self):
        """Save trained emotion detection model."""
        try:
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            model_file = self.model_path / "emotion_classifier.pkl"
            scaler_file = self.model_path / "feature_scaler.pkl"
            
            joblib.dump(self.model, model_file)
            joblib.dump(self.scaler, scaler_file)
            
            logger.info(f"Saved emotion model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save emotion model: {e}")
    
    def _create_default_model(self):
        """Create a default emotion detection model with synthetic data."""
        logger.info("Creating default emotion detection model with synthetic data")
        
        try:
            # Generate synthetic training data
            X, y = self._generate_synthetic_data()
            
            # Create and train model
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_scaled, y)
            
            # Save the model
            self._save_model()
            
            logger.info("Default emotion model created and saved")
            
        except Exception as e:
            logger.error(f"Failed to create default model: {e}")
            # Create a simple rule-based fallback
            self._create_rule_based_fallback()
    
    def _generate_synthetic_data(self, samples_per_emotion: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for emotion detection."""
        # This creates synthetic feature vectors based on typical patterns
        # In a real implementation, you would use actual labeled audio data
        
        X = []
        y = []
        
        # Define typical feature ranges for each emotion
        emotion_patterns = {
            'happy': {
                'pitch_mean': (200, 300),
                'pitch_std': (20, 40),
                'tempo': (120, 140),
                'rms_energy': (0.3, 0.7),
                'spectral_centroid': (2000, 4000),
                'zcr': (0.1, 0.3)
            },
            'sad': {
                'pitch_mean': (100, 180),
                'pitch_std': (10, 25),
                'tempo': (60, 90),
                'rms_energy': (0.1, 0.4),
                'spectral_centroid': (1000, 2500),
                'zcr': (0.05, 0.15)
            },
            'angry': {
                'pitch_mean': (180, 280),
                'pitch_std': (30, 50),
                'tempo': (100, 130),
                'rms_energy': (0.4, 0.8),
                'spectral_centroid': (2500, 5000),
                'zcr': (0.15, 0.35)
            },
            'anxious': {
                'pitch_mean': (160, 250),
                'pitch_std': (25, 45),
                'tempo': (90, 120),
                'rms_energy': (0.2, 0.6),
                'spectral_centroid': (1800, 3500),
                'zcr': (0.12, 0.28)
            },
            'calm': {
                'pitch_mean': (120, 200),
                'pitch_std': (8, 20),
                'tempo': (70, 100),
                'rms_energy': (0.15, 0.45),
                'spectral_centroid': (1200, 2800),
                'zcr': (0.06, 0.18)
            },
            'neutral': {
                'pitch_mean': (140, 220),
                'pitch_std': (15, 30),
                'tempo': (80, 110),
                'rms_energy': (0.2, 0.5),
                'spectral_centroid': (1500, 3000),
                'zcr': (0.08, 0.22)
            }
        }
        
        # Generate samples for each emotion
        for emotion in self.emotion_labels:
            if emotion not in emotion_patterns:
                continue
                
            pattern = emotion_patterns[emotion]
            
            for _ in range(samples_per_emotion):
                # Create a synthetic feature vector
                features = np.zeros(self.feature_extractor.config.mfcc_coefficients + 
                                  self.feature_extractor.config.chroma_features + 10)
                
                # MFCC features (synthetic)
                mfcc_base = np.random.normal(0, 1, self.feature_extractor.config.mfcc_coefficients)
                features[:self.feature_extractor.config.mfcc_coefficients] = mfcc_base
                
                # Chroma features (synthetic)
                chroma_start = self.feature_extractor.config.mfcc_coefficients
                chroma_end = chroma_start + self.feature_extractor.config.chroma_features
                features[chroma_start:chroma_end] = np.random.uniform(0, 1, 
                                                                   self.feature_extractor.config.chroma_features)
                
                # Prosodic and spectral features
                idx = chroma_end
                features[idx] = np.random.uniform(*pattern['spectral_centroid'])  # spectral_centroid
                features[idx + 1] = np.random.uniform(500, 2000)  # spectral_bandwidth
                features[idx + 2] = np.random.uniform(3000, 8000)  # spectral_rolloff
                features[idx + 3] = np.random.uniform(*pattern['zcr'])  # zero_crossing_rate
                features[idx + 4] = np.random.uniform(*pattern['rms_energy'])  # rms_energy
                features[idx + 5] = np.random.uniform(*pattern['tempo'])  # tempo
                features[idx + 6] = np.random.uniform(*pattern['pitch_mean'])  # pitch_mean
                features[idx + 7] = np.random.uniform(*pattern['pitch_std'])  # pitch_std
                features[idx + 8] = np.random.uniform(0, 0.1)  # jitter
                features[idx + 9] = np.random.uniform(0, 0.2)  # shimmer
                
                X.append(features)
                y.append(emotion)
        
        return np.array(X), np.array(y)
    
    def _create_rule_based_fallback(self):
        """Create a simple rule-based emotion detector as fallback."""
        logger.info("Creating rule-based emotion detection fallback")
        self.model = "rule_based"
        self.scaler = None
    
    def predict_emotion(self, audio: np.ndarray) -> EmotionPrediction:
        """Predict emotion from audio."""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(audio)
            feature_vector = self.feature_extractor.features_to_vector(features)
            
            if self.model == "rule_based":
                return self._rule_based_prediction(features)
            
            if self.model is None or self.scaler is None:
                return self._default_prediction()
            
            # Scale features
            feature_vector = feature_vector.reshape(1, -1)

            # Handle feature dimension mismatch
            expected_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else len(feature_vector[0])
            if len(feature_vector[0]) != expected_features:
                logger.warning(f"Feature dimension mismatch: got {len(feature_vector[0])}, expected {expected_features}")
                # Pad or truncate features to match expected size
                if len(feature_vector[0]) < expected_features:
                    # Pad with zeros
                    padding = np.zeros((1, expected_features - len(feature_vector[0])))
                    feature_vector = np.concatenate([feature_vector, padding], axis=1)
                else:
                    # Truncate
                    feature_vector = feature_vector[:, :expected_features]

            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Predict
            prediction = self.model.predict(feature_vector_scaled)[0]
            probabilities = self.model.predict_proba(feature_vector_scaled)[0]
            
            # Create probability dictionary
            prob_dict = {}
            for i, emotion in enumerate(self.emotion_labels):
                prob_dict[emotion] = float(probabilities[i])
            
            confidence = float(np.max(probabilities))
            
            # Apply confidence threshold
            if confidence < self.config.confidence_threshold:
                prediction = "neutral"
                confidence = prob_dict.get("neutral", 0.5)
            
            return EmotionPrediction(
                emotion=prediction,
                confidence=confidence,
                probabilities=prob_dict,
                features_used=len(feature_vector)
            )
            
        except Exception as e:
            logger.error(f"Emotion prediction failed: {e}")
            return self._default_prediction()
    
    def _rule_based_prediction(self, features: AudioFeatures) -> EmotionPrediction:
        """Simple rule-based emotion prediction."""
        # Simple heuristics based on audio features
        probabilities = {emotion: 0.0 for emotion in self.emotion_labels}
        
        # High energy and pitch -> happy or angry
        if features.rms_energy > 0.4 and features.pitch_mean > 200:
            if features.tempo > 110:
                probabilities['happy'] = 0.7
                probabilities['angry'] = 0.2
            else:
                probabilities['angry'] = 0.6
                probabilities['happy'] = 0.3
        
        # Low energy and pitch -> sad or calm
        elif features.rms_energy < 0.3 and features.pitch_mean < 180:
            if features.pitch_std < 20:
                probabilities['calm'] = 0.6
                probabilities['sad'] = 0.3
            else:
                probabilities['sad'] = 0.6
                probabilities['calm'] = 0.2
        
        # High pitch variation -> anxious
        elif features.pitch_std > 35:
            probabilities['anxious'] = 0.6
            probabilities['neutral'] = 0.3
        
        # Default to neutral
        else:
            probabilities['neutral'] = 0.8
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        else:
            probabilities['neutral'] = 1.0
        
        # Find emotion with highest probability
        emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[emotion]
        
        return EmotionPrediction(
            emotion=emotion,
            confidence=confidence,
            probabilities=probabilities,
            features_used=10  # Approximate number of features used
        )
    
    def _default_prediction(self) -> EmotionPrediction:
        """Return default neutral prediction."""
        probabilities = {emotion: 0.0 for emotion in self.emotion_labels}
        probabilities['neutral'] = 1.0
        
        return EmotionPrediction(
            emotion='neutral',
            confidence=0.5,
            probabilities=probabilities,
            features_used=0
        )
    
    def train_model(self, audio_files: List[str], labels: List[str]):
        """Train the emotion detection model with real data."""
        logger.info(f"Training emotion model with {len(audio_files)} samples")
        
        try:
            X = []
            y = []
            
            # Extract features from audio files
            for audio_file, label in zip(audio_files, labels):
                try:
                    audio, _ = self.feature_extractor.audio_processor.load_audio(audio_file)
                    features = self.feature_extractor.extract_features(audio)
                    feature_vector = self.feature_extractor.features_to_vector(features)
                    
                    X.append(feature_vector)
                    y.append(label)
                    
                except Exception as e:
                    logger.warning(f"Failed to process {audio_file}: {e}")
                    continue
            
            if len(X) == 0:
                logger.error("No valid training samples found")
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model trained with accuracy: {accuracy:.3f}")
            logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
            
            # Save model
            self._save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
