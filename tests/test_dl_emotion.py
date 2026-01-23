import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.emotion.detector import EmotionDetector

def test_dl_emotion():
    print("Testing Deep Learning Emotion Detector...")
    
    detector = EmotionDetector()
    print(f"Model Type: {detector.model_type}")
    
    if detector.model_type != 'wav2vec2':
        print("ERROR: Failed to load wav2vec2 model (fell back to random_forest)")
        return
        
    print("Model loaded successfully")
    
    # Create dummy audio (1 second of silence/noise)
    # Wav2Vec2 expects 16kHz
    sample_rate = 16000
    duration = 1.0 # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate a sine wave (just to have some signal)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio = audio.astype(np.float32)
    
    print("Running prediction on dummy audio...")
    prediction = detector.predict_emotion(audio)
    
    print(f"Prediction: {prediction.emotion}")
    print(f"Confidence: {prediction.confidence:.2f}")
    print(f"Probabilities: {prediction.probabilities}")
    
    if prediction.emotion in ['neutral', 'happy', 'sad', 'angry', 'anxious']:
        print("SUCCESS: Prediction returned valid emotion")
    else:
        print(f"ERROR: Invalid emotion returned: {prediction.emotion}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dl_emotion()
