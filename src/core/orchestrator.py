"""
Core streaming orchestrator for TUNDA.
Handles audio input, VAD, STT, LLM, and TTS in a full-duplex pipeline.
"""

import threading
import queue
import time
import logging
import numpy as np
import pyaudio
import collections
from typing import Optional, List, Callable

from ..utils.audio import AudioProcessor
from ..emotion.detector import EmotionDetector
from ..speech.recognition import SpeechRecognitionPipeline
from ..response.generator import EmpathicResponseGenerator, ResponseContext
from ..speech.synthesis import TextToSpeechPipeline
from ..memory.vector import VectorMemory
from ..memory.conversation import ConversationMemory
from ..utils.config import config

logger = logging.getLogger(__name__)

class StreamOrchestrator:
    def __init__(self):
        self.config = config
        self.running = False
        
        # Components
        self.sample_rate = self.config.audio.sample_rate
        self.chunk_size = self.config.audio.chunk_size
        self.channels = self.config.audio.channels
        self.audio_processor = AudioProcessor(sample_rate=self.sample_rate)
        self.emotion_detector = EmotionDetector()
        self.stt = SpeechRecognitionPipeline()
        self.llm = EmpathicResponseGenerator()
        self.tts = TextToSpeechPipeline()
        self.memory = VectorMemory()
        self.conversation_memory = ConversationMemory() if self.config.memory.enabled else None
        
        # Audio Input
        self.pyaudio_instance = pyaudio.PyAudio()
        self.input_stream = None
        
        # VAD & Buffering
        self.audio_buffer = collections.deque(maxlen=int(self.sample_rate * 30 / self.chunk_size)) # 30s max
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_counter = 0
        self.silence_threshold = self.config.audio.silence_threshold
        self.silence_threshold_chunks = int(self.config.audio.silence_duration * self.sample_rate / self.chunk_size)
        self.min_speech_seconds = self.config.get("audio.min_speech_seconds", 0.3)
        self.max_segment_seconds = self.config.get("audio.max_segment_seconds", 20.0)
        
        # Queues for pipeline
        queue_maxsize = self.config.get("audio.queue_maxsize", 10)
        self.transcription_queue = queue.Queue(maxsize=queue_maxsize)
        self.synthesis_queue = queue.Queue(maxsize=queue_maxsize)
        
        # Threads
        self.listen_thread = None
        self.process_thread = None
        self.speak_thread = None

    def start(self):
        """Start the streaming pipeline."""
        self.running = True
        logger.info("Starting Streaming Orchestrator...")
        if self.conversation_memory:
            self.conversation_memory.start_new_session()
        
        # Start Input Stream
        self.input_stream = self.pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.config.audio.input_device,
            stream_callback=self._audio_callback
        )
        self.input_stream.start_stream()
        
        # Start Threads
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        self.speak_thread = threading.Thread(target=self._speak_loop, daemon=True)
        self.speak_thread.start()
        
        logger.info("Orchestrator started.")

    def stop(self):
        """Stop the pipeline."""
        self.running = False
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.conversation_memory:
            self.conversation_memory.end_current_session()
        for thread in [self.process_thread, self.speak_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        self.pyaudio_instance.terminate()
        logger.info("Orchestrator stopped.")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for PyAudio input."""
        if not self.running:
            return (None, pyaudio.paComplete)
            
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        
        # Simple VAD (Energy based)
        rms = self.audio_processor.calculate_rms(audio_chunk)
        if rms > self.silence_threshold: # Threshold
            self.is_speaking = True
            self.silence_counter = 0
            self.speech_buffer.append(audio_chunk)
        else:
            if self.is_speaking:
                self.silence_counter += 1
                self.speech_buffer.append(audio_chunk) # Keep trailing silence
                
                if self.silence_counter > self.silence_threshold_chunks:
                    # Speech ended, push to processing
                    full_audio = np.concatenate(self.speech_buffer)
                    if len(full_audio) / self.sample_rate >= self.min_speech_seconds:
                        try:
                            self.transcription_queue.put(full_audio, timeout=0.2)
                        except queue.Full:
                            logger.warning("Transcription queue full. Dropping segment.")
                    self.speech_buffer = []
                    self.is_speaking = False
                    self.silence_counter = 0
        if self.is_speaking:
            if len(self.speech_buffer) * self.chunk_size / self.sample_rate > self.max_segment_seconds:
                full_audio = np.concatenate(self.speech_buffer)
                try:
                    self.transcription_queue.put(full_audio, timeout=0.2)
                except queue.Full:
                    logger.warning("Transcription queue full. Dropping long segment.")
                self.speech_buffer = []
                self.is_speaking = False
                self.silence_counter = 0
        
        return (in_data, pyaudio.paContinue)

    def _process_loop(self):
        """Consumes audio segments, transcribes, and generates streaming response."""
        while self.running:
            try:
                audio = self.transcription_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            logger.info(f"Processing speech segment: {len(audio)/self.sample_rate:.2f}s")
            if self.config.get("audio.noise_reduction", False):
                audio = self.audio_processor.apply_noise_reduction(audio)
            audio = self.audio_processor.normalize_audio(audio)
            audio = self.audio_processor.remove_silence(audio, threshold=self.silence_threshold)
            if len(audio) / self.sample_rate < self.min_speech_seconds:
                continue
            
            # 1. Transcribe
            try:
                transcription = self.stt.transcribe(audio)
            except Exception as exc:
                logger.error(f"Transcription failed: {exc}")
                continue
            text = transcription.text.strip()
            if not text:
                continue
            logger.info(f"User: {text}")
            
            # 2. Detect Emotion
            if self.config.emotion_detection.enabled:
                try:
                    emotion_pred = self.emotion_detector.predict_emotion(audio)
                except Exception as exc:
                    logger.error(f"Emotion detection failed: {exc}")
                    emotion_pred = None
            else:
                emotion_pred = None
            if emotion_pred is None:
                emotion_pred = self.emotion_detector._default_prediction()
            logger.info(f"Emotion: {emotion_pred.emotion} ({emotion_pred.confidence:.2f})")
            
            # 3. Retrieve Context & Generate Response (Streaming)
            retrieval_limit = self.config.get("memory.retrieval_limit", 3)
            relevant_memories = self.memory.retrieve_relevant(text, limit=retrieval_limit)
            conversation_history = []
            user_name = "Friend"
            user_preferences = {}
            if self.conversation_memory:
                conversation_history = self.conversation_memory.get_conversation_context()
                user_name = self.conversation_memory.get_user_name() or user_name
                user_preferences = self.conversation_memory.get_user_preferences()
            user_preferences = {**user_preferences, "user_name": user_name}
            
            context = ResponseContext(
                user_text=text,
                emotion=emotion_pred.emotion,
                confidence=emotion_pred.confidence,
                conversation_history=conversation_history,
                empathy_style=config.response_generation.default_style,
                user_preferences=user_preferences,
                relevant_memories=relevant_memories
            )
            
            token_stream = self.llm.generate_response_stream(context)
            
            # 4. Buffer Sentences for TTS & Accumulate for Memory
            current_sentence = ""
            full_response = ""
            
            for token in token_stream:
                full_response += token
                current_sentence += token
                if any(punct in token for punct in ['.', '!', '?']):
                    # Send sentence to synthesis
                    self.synthesis_queue.put((current_sentence, emotion_pred.emotion))
                    current_sentence = ""
            
            if current_sentence.strip():
                try:
                    self.synthesis_queue.put((current_sentence, emotion_pred.emotion), timeout=0.2)
                except queue.Full:
                    logger.warning("Synthesis queue full. Dropping tail sentence.")
                
            # 5. Save Interaction to Long-term Memory
            full_response = full_response.strip()
            if full_response:
                self.memory.add_memory(
                    text=f"User: {text}\nAssistant: {full_response}",
                    metadata={"emotion": emotion_pred.emotion}
                )
                if self.conversation_memory:
                    self.conversation_memory.add_conversation_turn(
                        user_text=text,
                        user_emotion=emotion_pred.emotion,
                        user_confidence=emotion_pred.confidence,
                        assistant_response=full_response,
                        empathy_style=context.empathy_style,
                        response_confidence=0.8
                    )

    def _speak_loop(self):
        """Consumes text sentences and plays them immediately."""
        while self.running:
            try:
                text, emotion = self.synthesis_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            logger.info(f"Synthesizing: {text}")
            if not text.strip():
                continue
            
            # Synthesize
            # TTS pipeline currently synthesizes to memory. 
            # We ideally need to stream the audio out.
            # For now, we synthesize the whole sentence and play it.
            result = self.tts.synthesize(text, emotion=emotion)
            
            if result.success and len(result.audio) > 0:
                self._play_audio(result.audio, result.sample_rate)

    def _play_audio(self, audio: np.ndarray, sample_rate: int):
        """Play audio array."""
        # Open output stream just for this clip (simple but not truly streaming)
        # Using a proper output stream would be better for back-to-back sentences.
        output_stream = self.pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            output=True,
            output_device_index=self.config.audio.output_device
        )
        output_stream.write(audio.astype(np.float32).tobytes())
        output_stream.stop_stream()
        output_stream.close()
