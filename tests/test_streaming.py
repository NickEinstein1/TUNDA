
import sys
import os
import time
import numpy as np
import logging
import threading

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.orchestrator import StreamOrchestrator

def test_streaming():
    print("Testing Streaming Orchestrator...")
    
    orchestrator = StreamOrchestrator()
    
    # Mock PyAudio to avoid needing a microphone in CI/Headless env
    # But for a local user test, we can try to run it for real for X seconds
    # OR we can inject audio into the orchestrator's queues manually.
    
    print("Starting orchestrator (runs for 10 seconds)...")
    try:
        orchestrator.start()
        
        # Simulate an audio injection (pretend we heard "Hello, I am sad.")
        # We'll assume 16kHz sample rate
        sr = 16000
        
        # Inject into transcription queue directly to test the processing pipeline
        # (Skipping audio input -> VAD logic for this specific sub-test)
        print("Injecting dummy audio (simulating 'Hello')...")
        dummy_audio = np.zeros(int(sr * 2.0), dtype=np.float32) # Silence, but pipeline will process it
        orchestrator.transcription_queue.put(dummy_audio)
        
        time.sleep(10)
        
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.stop()
        print("Orchestrator stopped.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_streaming()
