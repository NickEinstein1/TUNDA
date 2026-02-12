#!/usr/bin/env python3
"""
Empathic Voice Companion - Main Application
Streamlined version using the new Streaming Architecture.
"""

import sys
import logging
import signal
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import config
from src.core.orchestrator import StreamOrchestrator
from src.core.async_orchestrator import AsyncStreamOrchestrator

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

def main():
    """Main entry point."""
    print("\n🚀 Starting TUNDA (Streaming Mode)...")
    print("🎤 Listening for voice input...")
    print("🛑 Press Ctrl+C to stop\n")
    mode = config.get("orchestrator.mode", "threaded")
    orchestrator = AsyncStreamOrchestrator() if mode == "async" else StreamOrchestrator()
    
    def signal_handler(signum, frame):
        print("\n👋 Stopping...")
        if mode == "async":
            orchestrator.stop_sync()
        else:
            orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if mode == "async":
            import asyncio
            async def runner():
                await orchestrator.start()
                while True:
                    await asyncio.sleep(1.0)
            asyncio.run(runner())
        else:
            orchestrator.start()
            # Keep main thread alive
            while True:
                time.sleep(1.0)
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error: {e}")
        orchestrator.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()
