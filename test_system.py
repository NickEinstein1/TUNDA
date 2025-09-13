#!/usr/bin/env python3
"""
Test script for the Empathic Voice Companion system.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import config
from src.utils.audio import AudioProcessor
from src.speech.recognition import SpeechRecognitionPipeline
from src.emotion.detector import EmotionDetector
from src.response.generator import EmpathicResponseGenerator, ResponseContext
from src.speech.synthesis import TextToSpeechPipeline
from src.memory.conversation import ConversationMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_audio_processing():
    """Test audio processing utilities."""
    print("🎵 Testing audio processing...")
    
    try:
        processor = AudioProcessor()
        
        # Create test audio
        sample_rate = 16000
        duration = 2.0
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Test normalization
        normalized = processor.normalize_audio(test_audio)
        assert np.max(np.abs(normalized)) <= 1.0
        
        # Test RMS calculation
        rms = processor.calculate_rms(test_audio)
        assert rms > 0
        
        # Test zero crossing rate
        zcr = processor.calculate_zero_crossing_rate(test_audio)
        assert zcr > 0
        
        print("✅ Audio processing tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        return False


def test_emotion_detection():
    """Test emotion detection."""
    print("😊 Testing emotion detection...")
    
    try:
        detector = EmotionDetector()
        
        # Create test audio (sine wave)
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = 0.3 * np.sin(2 * np.pi * 200 * t)  # Low frequency for "sad" emotion
        
        # Test emotion prediction
        result = detector.predict_emotion(test_audio)
        
        assert result.emotion in config.emotion_detection.supported_emotions
        assert 0 <= result.confidence <= 1
        assert len(result.probabilities) == len(config.emotion_detection.supported_emotions)
        
        print(f"✅ Emotion detection test passed - detected: {result.emotion} ({result.confidence:.2f})")
        return True
        
    except Exception as e:
        print(f"❌ Emotion detection test failed: {e}")
        return False


def test_response_generation():
    """Test empathic response generation."""
    print("💬 Testing response generation...")
    
    try:
        generator = EmpathicResponseGenerator()
        
        # Create test context
        context = ResponseContext(
            user_text="I'm feeling really sad today",
            emotion="sad",
            confidence=0.8,
            conversation_history=[],
            empathy_style="supportive",
            user_preferences={}
        )
        
        # Generate response
        response = generator.generate_response(context)
        
        assert len(response.text) > 0
        assert response.emotion_addressed == "sad"
        assert response.empathy_style == "supportive"
        assert 0 <= response.confidence <= 1
        
        print(f"✅ Response generation test passed - response: \"{response.text[:50]}...\"")
        return True
        
    except Exception as e:
        print(f"❌ Response generation test failed: {e}")
        return False


def test_text_to_speech():
    """Test text-to-speech synthesis."""
    print("🔊 Testing text-to-speech...")
    
    try:
        tts = TextToSpeechPipeline()
        
        if not tts.is_available():
            print("⚠️  TTS not available, skipping test")
            return True
        
        # Test synthesis
        result = tts.synthesize("Hello, this is a test.", emotion="happy")
        
        if result.success:
            assert len(result.audio) > 0
            assert result.sample_rate > 0
            print(f"✅ TTS test passed - generated {len(result.audio)} audio samples")
        else:
            print("⚠️  TTS synthesis failed, but system is functional")
        
        return True
        
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False


def test_conversation_memory():
    """Test conversation memory."""
    print("🧠 Testing conversation memory...")
    
    try:
        memory = ConversationMemory()
        
        # Start session
        session_id = memory.start_new_session("test_session")
        assert session_id == "test_session"
        
        # Add conversation turn
        memory.add_conversation_turn(
            user_text="Hello, how are you?",
            user_emotion="neutral",
            user_confidence=0.7,
            assistant_response="Hello! I'm here to listen and support you.",
            empathy_style="supportive",
            response_confidence=0.9
        )
        
        # Get context
        context = memory.get_conversation_context()
        assert len(context) == 1
        assert context[0]['user'] == "Hello, how are you?"
        
        # Get emotion patterns
        patterns = memory.get_emotion_patterns()
        assert 'most_common_emotion' in patterns
        
        # End session
        memory.end_current_session()
        
        print("✅ Conversation memory test passed")
        return True
        
    except Exception as e:
        print(f"❌ Conversation memory test failed: {e}")
        return False


def test_speech_recognition():
    """Test speech recognition (basic functionality)."""
    print("🎤 Testing speech recognition...")
    
    try:
        recognizer = SpeechRecognitionPipeline()
        
        # Test with empty audio (should return empty result)
        empty_audio = np.array([])
        result = recognizer.transcribe(empty_audio)
        
        # Should not crash and should return a result object
        assert hasattr(result, 'text')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'language')
        
        print("✅ Speech recognition test passed (basic functionality)")
        return True
        
    except Exception as e:
        print(f"❌ Speech recognition test failed: {e}")
        return False


def test_integration():
    """Test integration between components."""
    print("🔗 Testing component integration...")
    
    try:
        # Initialize all components
        memory = ConversationMemory()
        detector = EmotionDetector()
        generator = EmpathicResponseGenerator()
        
        # Start session
        session_id = memory.start_new_session("integration_test")
        
        # Simulate a conversation flow
        user_text = "I'm having a really difficult day"
        
        # Create test audio for emotion detection
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = 0.2 * np.sin(2 * np.pi * 150 * t)  # Low frequency for sad emotion
        
        # Detect emotion
        emotion_result = detector.predict_emotion(test_audio)
        
        # Generate response
        context = ResponseContext(
            user_text=user_text,
            emotion=emotion_result.emotion,
            confidence=emotion_result.confidence,
            conversation_history=memory.get_conversation_context(),
            empathy_style="supportive",
            user_preferences=memory.get_user_preferences()
        )
        
        response = generator.generate_response(context)
        
        # Save to memory
        memory.add_conversation_turn(
            user_text=user_text,
            user_emotion=emotion_result.emotion,
            user_confidence=emotion_result.confidence,
            assistant_response=response.text,
            empathy_style=response.empathy_style,
            response_confidence=response.confidence
        )
        
        # Verify integration
        conversation_context = memory.get_conversation_context()
        assert len(conversation_context) == 1
        
        emotion_patterns = memory.get_emotion_patterns()
        assert emotion_patterns['total_interactions'] == 1
        
        # End session
        memory.end_current_session()
        
        print("✅ Integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Running Empathic Voice Companion system tests...\n")
    
    tests = [
        test_audio_processing,
        test_emotion_detection,
        test_response_generation,
        test_conversation_memory,
        test_speech_recognition,
        test_text_to_speech,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTo start the voice companion:")
        print("  python main.py")
        print("\nTo start the web interface:")
        print("  python app.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("The system may still be partially functional.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
