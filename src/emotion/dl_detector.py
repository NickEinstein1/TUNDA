"""Deep Learning based emotion detection using Wav2Vec2."""

import logging
import numpy as np
import torch
from typing import Dict, Optional, Union
import time
from transformers import pipeline

from .types import EmotionPrediction
from ..utils.config import config

logger = logging.getLogger(__name__)

class DeepLearningEmotionDetector:
    """Emotion detector using HuggingFace Transformers (Wav2Vec2)."""
    
    def __init__(self):
        self.config = config.emotion_detection
        self.model_name = getattr(self.config, "dl_model_name", "superb/wav2vec2-base-superb-er")
        self.device = self.config.device
        self.pipeline = None
        self.emotion_map = {
            "neu": "neutral",
            "hap": "happy",
            "sad": "sad",
            "ang": "angry",
            "fear": "anxious", # Mapping fear to anxious
        }
        self._load_model()

    def _load_model(self):
        """Load the Wav2Vec2 model pipeline."""
        try:
            logger.info(f"Loading emotion model: {self.model_name}")
            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline = pipeline(
                "audio-classification", 
                model=self.model_name,
                device=0 if device == "cuda" else -1
            )
            logger.info("Deep Learning emotion model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DL emotion model: {e}")
            self.pipeline = None

    def predict_emotion(self, audio: np.ndarray) -> EmotionPrediction:
        """Predict emotion from raw audio waveform."""
        if self.pipeline is None:
            return self._default_prediction()

        try:
            start_time = time.time()
            
            # Transformers pipeline expects numpy array
            # Ensure audio is float32 and correct shape
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Run inference
            # usage of pipeline with raw audio: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AudioClassificationPipeline
            # It usually expects a file path or a dict with "array" and "sampling_rate"
            # We assume 16kHz sample rate as per Wav2Vec2 standard, or we verify config
            
            # Note: models usually trained on 16kHz
            # If our system uses different rate, we might need resampling. 
            # Assuming standard 16000Hz for now as many AS models use it.
            # config.audio.sample_rate should be checked.
            
            predictions = self.pipeline(audio) # Check if this accepts raw numpy array directly in this version, usually yes
            
            # predictions is a list of dicts: [{'score': 0.9, 'label': 'neu'}, ...]
            if not predictions:
                return self._default_prediction()

            # Inspect top prediction
            top_pred = predictions[0]
            label = top_pred['label']
            score = top_pred['score']
            
            # Map label to TUNDA emotions
            mapped_emotion = self.emotion_map.get(label, "neutral")
            
            # Build probabilities dict
            probabilities = {}
            for pred in predictions:
                mapped_label = self.emotion_map.get(pred['label'], "other")
                if mapped_label != "other":
                     # If multiple labels map to same emotion, take max or sum? 
                     # Wav2Vec2-ER classes are usually exclusive (neu, hap, sad, ang)
                    probabilities[mapped_label] = pred['score']

            return EmotionPrediction(
                emotion=mapped_emotion,
                confidence=float(score),
                probabilities=probabilities,
                features_used=0 # Not using manual features
            )

        except Exception as e:
            logger.error(f"DL Emotion prediction failed: {e}")
            return self._default_prediction()

    def _default_prediction(self) -> EmotionPrediction:
        """Return neutral fallback."""
        return EmotionPrediction(
            emotion="neutral",
            confidence=0.0,
            probabilities={"neutral": 1.0},
            features_used=0
        )
