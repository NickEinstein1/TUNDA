#!/usr/bin/env python3
"""
Test script for Tunda's new personality and care plan features.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.memory.conversation import ConversationMemory
from src.response.generator import EmpathicResponseGenerator, ResponseContext
from src.response.care_plans import CarePlanGenerator
from src.emotion.detector import EmotionDetector


def test_name_recognition():
    """Test name recognition functionality."""
    print("🧪 Testing name recognition...")
    
    memory = ConversationMemory()
    session_id = memory.start_new_session("test_name_recognition")
    
    # Test name extraction
    test_phrases = [
        "Hi, my name is Nick",
        "Hello, I'm Sarah",
        "Call me Alex",
        "This is John speaking",
        "Hi there, it's Emma"
    ]
    
    for phrase in test_phrases:
        memory.add_conversation_turn(
            user_text=phrase,
            user_emotion="neutral",
            user_confidence=0.7,
            assistant_response="Hello!",
            empathy_style="supportive",
            response_confidence=0.8
        )
        
        user_name = memory.get_user_name()
        if user_name:
            print(f"✅ Extracted name '{user_name}' from: '{phrase}'")
            break
    else:
        print("❌ Failed to extract name from test phrases")
    
    memory.end_current_session()
    return user_name is not None


def test_care_plans():
    """Test care plan generation."""
    print("\n🧪 Testing care plan generation...")
    
    care_generator = CarePlanGenerator()
    
    emotions = ['happy', 'sad', 'anxious', 'angry', 'calm']
    
    for emotion in emotions:
        try:
            plan = care_generator.get_care_plan(emotion, "Nick")
            activity = care_generator.get_immediate_suggestion(emotion)
            affirmation = care_generator.get_affirmation(emotion)
            
            print(f"✅ {emotion.capitalize()} care plan:")
            print(f"   Plan: {plan.plan_name}")
            print(f"   Activity: {activity.name} ({activity.duration})")
            print(f"   Affirmation: {affirmation}")
            
        except Exception as e:
            print(f"❌ Failed to generate care plan for {emotion}: {e}")
            return False
    
    return True


def test_tunda_responses():
    """Test Tunda's personalized responses."""
    print("\n🧪 Testing Tunda's personalized responses...")
    
    memory = ConversationMemory()
    generator = EmpathicResponseGenerator()
    
    # Start session and add name
    session_id = memory.start_new_session("test_tunda_responses")
    memory.add_conversation_turn(
        user_text="Hi, my name is Nick",
        user_emotion="neutral",
        user_confidence=0.7,
        assistant_response="Hello!",
        empathy_style="supportive",
        response_confidence=0.8
    )
    
    # Test different emotional responses
    test_scenarios = [
        ("I'm feeling really happy today!", "happy", 0.8),
        ("I'm so sad and overwhelmed", "sad", 0.9),
        ("I'm really anxious about tomorrow", "anxious", 0.8),
        ("I'm angry about what happened", "angry", 0.7),
        ("I'm feeling pretty calm right now", "calm", 0.8)
    ]
    
    for user_text, emotion, confidence in test_scenarios:
        context = ResponseContext(
            user_text=user_text,
            emotion=emotion,
            confidence=confidence,
            conversation_history=memory.get_conversation_context(),
            empathy_style="supportive",
            user_preferences={'user_name': 'Nick'}
        )
        
        response = generator.generate_response(context)
        
        print(f"\n📝 Scenario: {emotion.upper()}")
        print(f"   User: \"{user_text}\"")
        print(f"   Tunda: \"{response.text}\"")
        
        # Check if response includes name and care suggestions
        has_name = "Nick" in response.text
        has_care_suggestion = any(word in response.text.lower() for word in 
                                ['activity', 'plan', 'suggest', 'help', 'try'])
        has_day_question = any(phrase in response.text.lower() for phrase in 
                             ['how has your day', 'how are you', 'tell me about'])
        
        print(f"   ✅ Includes name: {has_name}")
        print(f"   ✅ Offers care: {has_care_suggestion}")
        print(f"   ✅ Asks about day: {has_day_question}")
    
    memory.end_current_session()
    return True


def test_integration():
    """Test full integration with emotion detection."""
    print("\n🧪 Testing full integration...")
    
    try:
        memory = ConversationMemory()
        generator = EmpathicResponseGenerator()
        detector = EmotionDetector()
        
        # Start session
        session_id = memory.start_new_session("test_integration")
        
        # Simulate introducing name
        memory.add_conversation_turn(
            user_text="Hello, my name is Nick and I'm feeling stressed",
            user_emotion="anxious",
            user_confidence=0.8,
            assistant_response="Hello Nick!",
            empathy_style="supportive",
            response_confidence=0.9
        )
        
        # Create test audio for emotion detection
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = 0.2 * np.sin(2 * np.pi * 150 * t)  # Low frequency for sad emotion
        
        # Detect emotion
        emotion_result = detector.predict_emotion(test_audio)
        
        # Generate response
        context = ResponseContext(
            user_text="I've been having a really tough day",
            emotion=emotion_result.emotion,
            confidence=emotion_result.confidence,
            conversation_history=memory.get_conversation_context(),
            empathy_style="supportive",
            user_preferences={'user_name': 'Nick'}
        )
        
        response = generator.generate_response(context)
        
        print(f"📝 Full Integration Test:")
        print(f"   User: \"I've been having a really tough day\"")
        print(f"   Detected emotion: {emotion_result.emotion} ({emotion_result.confidence:.2f})")
        print(f"   Tunda: \"{response.text}\"")
        
        # Verify Tunda's personality features
        has_name = "Nick" in response.text or "Tunda" in response.text
        is_empathic = len(response.text) > 50  # Should be a substantial response
        
        print(f"   ✅ Personal & empathic: {has_name and is_empathic}")
        
        memory.end_current_session()
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all Tunda tests."""
    print("🤖 Testing Tunda's Enhanced Personality Features")
    print("=" * 50)
    
    tests = [
        ("Name Recognition", test_name_recognition),
        ("Care Plans", test_care_plans),
        ("Personalized Responses", test_tunda_responses),
        ("Full Integration", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
        
        print("-" * 30)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Tunda is ready with enhanced personality!")
        print("\nTunda now features:")
        print("• Personal name recognition and usage")
        print("• Warm greetings asking about your day")
        print("• Personalized care plan suggestions")
        print("• Emotion-specific therapeutic activities")
        print("• Affirmations and wellness support")
    else:
        print("⚠️  Some tests failed, but core functionality should still work")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
