import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.speech.recognition import SpeechRecognitionPipeline


class DummyRecognizer:
    def __init__(self, name):
        self.name = name
        self.model = True

    def transcribe(self, *args, **kwargs):
        return type("Result", (), {"text": "ok", "confidence": 1.0, "language": "en", "segments": [], "processing_time": 0.0})


def test_dynamic_model_switching(monkeypatch):
    def fake_create(self, model_name, device):
        return DummyRecognizer(model_name)

    monkeypatch.setattr(SpeechRecognitionPipeline, "_resolve_device", lambda self: "cuda")
    monkeypatch.setattr(SpeechRecognitionPipeline, "_has_large_stt_budget", lambda self: True)
    monkeypatch.setattr(SpeechRecognitionPipeline, "_create_recognizer", fake_create)

    pipeline = SpeechRecognitionPipeline()
    short_audio = np.zeros(16000, dtype=np.float32)  # 1s
    long_audio = np.zeros(16000 * 10, dtype=np.float32)  # 10s

    short_rec = pipeline._choose_recognizer(short_audio)
    long_rec = pipeline._choose_recognizer(long_audio)

    assert short_rec.name == pipeline.config.short_model
    assert long_rec.name == pipeline.config.long_model
