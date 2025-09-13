#!/usr/bin/env python3
"""
Empathic Voice Companion - Main Application

A voice AI system that detects emotional states from speech and responds with appropriate empathy.
"""

import sys
import logging
import signal
import time
import threading
from pathlib import Path
import numpy as np
import pyaudio
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import config
from src.utils.audio import AudioProcessor, AudioBuffer
from src.speech.recognition import SpeechRecognitionPipeline
from src.emotion.detector import EmotionDetector
from src.response.generator import EmpathicResponseGenerator, ResponseContext
from src.speech.synthesis import TextToSpeechPipeline
from src.memory.conversation import ConversationMemory

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.get('logging.level', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.get('logging.file', 'logs/empathic_voice.log'))
    ]
)

logger = logging.getLogger(__name__)


class EmpathicVoiceCompanion:
    """Main application class for the Empathic Voice Companion."""
    
    def __init__(self):
        self.running = False
        self.audio_stream = None
        self.audio = None
        
        # Initialize components
        logger.info("Initializing Empathic Voice Companion...")
        
        try:
            self.audio_processor = AudioProcessor(sample_rate=config.audio.sample_rate)
            self.audio_buffer = AudioBuffer(max_duration=10.0, sample_rate=config.audio.sample_rate)
            self.speech_recognizer = SpeechRecognitionPipeline()
            self.emotion_detector = EmotionDetector()
            self.response_generator = EmpathicResponseGenerator()
            self.tts_pipeline = TextToSpeechPipeline()
            self.conversation_memory = ConversationMemory()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def start(self):
        """Start the voice companion."""
        logger.info("Starting Empathic Voice Companion")
        
        # Start conversation session
        session_id = self.conversation_memory.start_new_session()
        logger.info(f"Started conversation session: {session_id}")
        
        # Check component availability
        self._check_component_availability()
        
        # Initialize audio
        self._initialize_audio()
        
        # Start main loop
        self.running = True
        self._main_loop()
    
    def stop(self):
        """Stop the voice companion."""
        logger.info("Stopping Empathic Voice Companion")
        self.running = False
        
        # Stop audio stream
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        # End conversation session
        self.conversation_memory.end_current_session()
        
        logger.info("Empathic Voice Companion stopped")
    
    def _check_component_availability(self):
        """Check availability of all components."""
        logger.info("Checking component availability...")
        
        # Check TTS
        if not self.tts_pipeline.is_available():
            logger.warning("Text-to-Speech not available - responses will be text-only")
        
        # Check LLM
        if not self.response_generator.is_llm_available():
            logger.warning("LLM not available - using template-based responses")
        
        # Check supported languages
        languages = self.speech_recognizer.get_supported_languages()
        logger.info(f"Speech recognition supports {len(languages)} languages")
    
    def _initialize_audio(self):
        """Initialize audio input/output."""
        try:
            self.audio = pyaudio.PyAudio()
            
            # Get audio device info
            input_device = config.audio.input_device
            if input_device is None:
                input_device = self.audio.get_default_input_device_info()['index']
            
            logger.info(f"Using audio input device: {input_device}")
            
            # Create audio stream
            self.audio_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=config.audio.channels,
                rate=config.audio.sample_rate,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=config.audio.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.audio_stream.start_stream()
            logger.info("Audio stream started")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio input callback."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert audio data
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to buffer
        self.audio_buffer.write(audio_data)
        
        return (None, pyaudio.paContinue)
    
    def _main_loop(self):
        """Main processing loop."""
        logger.info("Starting main processing loop")
        
        last_processing_time = time.time()
        silence_start = None
        
        print("\n🎤 Empathic Voice Companion is listening...")
        print("💬 Speak naturally, and I'll respond with empathy")
        print("🛑 Press Ctrl+C to stop\n")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Process audio every 100ms
                if current_time - last_processing_time >= 0.1:
                    self._process_audio_chunk()
                    last_processing_time = current_time
                
                # Check for silence
                recent_audio = self.audio_buffer.read(1.0)  # Last 1 second
                
                if len(recent_audio) > 0:
                    rms = self.audio_processor.calculate_rms(recent_audio)
                    
                    if rms > config.audio.silence_threshold:
                        # Voice detected
                        silence_start = None
                    else:
                        # Silence detected
                        if silence_start is None:
                            silence_start = current_time
                        elif current_time - silence_start >= config.audio.silence_duration:
                            # Process accumulated audio
                            self._process_speech_segment()
                            silence_start = None
                
                time.sleep(0.05)  # Small sleep to prevent busy waiting
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)  # Prevent rapid error loops
    
    def _process_audio_chunk(self):
        """Process a small chunk of audio for real-time feedback."""
        # This could be used for real-time voice activity detection
        # or other real-time processing features
        pass
    
    def _process_speech_segment(self):
        """Process a complete speech segment."""
        try:
            # Get audio from buffer
            audio_segment = self.audio_buffer.read(5.0)  # Last 5 seconds
            
            if len(audio_segment) == 0:
                return
            
            # Check if speech is detected
            if not self.speech_recognizer.is_speech_detected(audio_segment):
                return
            
            print("🎯 Processing speech...")
            
            # Speech recognition
            transcription_result = self.speech_recognizer.transcribe(audio_segment)
            
            if not transcription_result.text.strip():
                return
            
            user_text = transcription_result.text.strip()
            print(f"👤 You said: \"{user_text}\"")
            
            # Emotion detection
            emotion_result = self.emotion_detector.predict_emotion(audio_segment)
            print(f"😊 Detected emotion: {emotion_result.emotion} (confidence: {emotion_result.confidence:.2f})")
            
            # Save to conversation memory first to extract name
            self.conversation_memory.add_conversation_turn(
                user_text=user_text,
                user_emotion=emotion_result.emotion,
                user_confidence=emotion_result.confidence,
                assistant_response="",  # Will be updated after generation
                empathy_style=config.response_generation.default_style,
                response_confidence=0.0  # Will be updated after generation
            )

            # Generate empathic response
            user_name = self.conversation_memory.get_user_name()
            response_context = ResponseContext(
                user_text=user_text,
                emotion=emotion_result.emotion,
                confidence=emotion_result.confidence,
                conversation_history=self.conversation_memory.get_conversation_context(),
                empathy_style=config.response_generation.default_style,
                user_preferences=self.conversation_memory.get_user_preferences()
            )

            # Add user name to preferences if known
            if user_name:
                response_context.user_preferences['user_name'] = user_name
            
            empathic_response = self.response_generator.generate_response(response_context)
            print(f"🤖 Response: \"{empathic_response.text}\"")
            
            # Text-to-speech
            if self.tts_pipeline.is_available():
                print("🔊 Generating speech...")
                tts_result = self.tts_pipeline.synthesize(
                    empathic_response.text, 
                    emotion=emotion_result.emotion
                )
                
                if tts_result.success:
                    self._play_audio(tts_result.audio, tts_result.sample_rate)
                    print("✅ Speech played")
                else:
                    print("❌ Speech synthesis failed")
            
            # Update the last conversation turn with the response
            if self.conversation_memory.current_session and self.conversation_memory.current_session.turns:
                last_turn = self.conversation_memory.current_session.turns[-1]
                last_turn.assistant_response = empathic_response.text
                last_turn.empathy_style = empathic_response.empathy_style
                last_turn.response_confidence = empathic_response.confidence
            
            # Show emotion patterns
            emotion_patterns = self.conversation_memory.get_emotion_patterns()
            if emotion_patterns.get('total_interactions', 0) > 1:
                print(f"📊 Emotion trend: {emotion_patterns.get('recent_trend', 'stable')}")
            
            print("-" * 50)
            
        except Exception as e:
            logger.error(f"Error processing speech segment: {e}")
            print(f"❌ Error processing speech: {e}")
    
    def _play_audio(self, audio: np.ndarray, sample_rate: int):
        """Play audio through speakers."""
        try:
            # Resample if necessary
            if sample_rate != config.audio.sample_rate:
                audio = self.audio_processor.resample_audio(
                    audio, sample_rate, config.audio.sample_rate
                )
            
            # Normalize audio
            audio = self.audio_processor.normalize_audio(audio)
            
            # Convert to bytes
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            
            # Create output stream
            output_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=config.audio.sample_rate,
                output=True
            )
            
            # Play audio
            output_stream.write(audio_bytes)
            output_stream.stop_stream()
            output_stream.close()
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    def print_statistics(self):
        """Print session statistics."""
        stats = self.conversation_memory.get_session_statistics()
        
        if not stats:
            print("No conversation data available.")
            return
        
        print("\n📈 Session Statistics:")
        print(f"   Total interactions: {stats.get('total_turns', 0)}")
        print(f"   Average emotion confidence: {stats.get('average_emotion_confidence', 0):.2f}")
        print(f"   Average response confidence: {stats.get('average_response_confidence', 0):.2f}")
        
        emotion_patterns = stats.get('emotion_patterns', {})
        if emotion_patterns:
            print(f"   Most common emotion: {emotion_patterns.get('most_common_emotion', 'unknown')}")
            print(f"   Emotional trend: {emotion_patterns.get('recent_trend', 'stable')}")


def main():
    """Main entry point."""
    try:
        # Create and start the voice companion
        companion = EmpathicVoiceCompanion()
        companion.start()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        sys.exit(1)
    finally:
        # Print final statistics
        if 'companion' in locals():
            companion.print_statistics()


if __name__ == "__main__":
    main()
