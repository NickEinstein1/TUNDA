#!/usr/bin/env python3
"""
Setup script for downloading and configuring models for the Empathic Voice Companion.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import tarfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ModelSetup:
    """Setup and download models for the Empathic Voice Companion."""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
    
    def setup_all(self):
        """Setup all required models and dependencies."""
        print("🚀 Setting up Empathic Voice Companion models...")
        print("This may take a while depending on your internet connection.\n")
        
        # Check Python version
        self.check_python_version()
        
        # Install system dependencies
        self.install_system_dependencies()
        
        # Setup Whisper models
        self.setup_whisper_models()
        
        # Setup Piper TTS models
        self.setup_piper_tts()
        
        # Setup Ollama (optional)
        self.setup_ollama()
        
        # Create directories
        self.create_directories()
        
        print("\n✅ Setup complete!")
        print("You can now run the Empathic Voice Companion with: python main.py")
    
    def check_python_version(self):
        """Check Python version compatibility."""
        print("🐍 Checking Python version...")
        
        if sys.version_info < (3, 9):
            print("❌ Python 3.9 or higher is required")
            sys.exit(1)
        
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    def install_system_dependencies(self):
        """Install system-level dependencies."""
        print("\n📦 Installing system dependencies...")
        
        system = os.name
        
        if system == 'nt':  # Windows
            self.install_windows_dependencies()
        elif system == 'posix':  # Linux/macOS
            self.install_unix_dependencies()
        else:
            print("⚠️  Unknown system, skipping system dependencies")
    
    def install_windows_dependencies(self):
        """Install Windows-specific dependencies."""
        print("🪟 Detected Windows system")
        
        # Check for FFmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            print("✅ FFmpeg is already installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ FFmpeg not found. Please install FFmpeg manually:")
            print("   1. Download from https://ffmpeg.org/download.html")
            print("   2. Add to your PATH environment variable")
        
        # Install PyAudio wheel if needed
        try:
            import pyaudio
            print("✅ PyAudio is available")
        except ImportError:
            print("📥 Installing PyAudio...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "pyaudio"], check=True)
                print("✅ PyAudio installed")
            except subprocess.CalledProcessError:
                print("❌ Failed to install PyAudio. You may need to install it manually.")
    
    def install_unix_dependencies(self):
        """Install Unix-specific dependencies."""
        print("🐧 Detected Unix-like system")
        
        # Check for package managers and install dependencies
        if self.command_exists("apt-get"):  # Ubuntu/Debian
            print("📥 Installing dependencies with apt...")
            deps = ["ffmpeg", "portaudio19-dev", "python3-pyaudio", "espeak", "espeak-data"]
            for dep in deps:
                try:
                    subprocess.run(["sudo", "apt-get", "install", "-y", dep], 
                                 capture_output=True, check=True)
                except subprocess.CalledProcessError:
                    print(f"⚠️  Failed to install {dep}")
        
        elif self.command_exists("brew"):  # macOS
            print("📥 Installing dependencies with Homebrew...")
            deps = ["ffmpeg", "portaudio"]
            for dep in deps:
                try:
                    subprocess.run(["brew", "install", dep], 
                                 capture_output=True, check=True)
                except subprocess.CalledProcessError:
                    print(f"⚠️  Failed to install {dep}")
        
        else:
            print("⚠️  No supported package manager found")
            print("Please install manually: ffmpeg, portaudio, espeak")
    
    def setup_whisper_models(self):
        """Setup Whisper models."""
        print("\n🎤 Setting up Whisper models...")
        
        whisper_dir = self.models_dir / "whisper"
        whisper_dir.mkdir(exist_ok=True)
        
        # Whisper models will be downloaded automatically on first use
        print("✅ Whisper models will be downloaded automatically on first use")
        
        # Test Whisper installation
        try:
            import whisper
            print("✅ Whisper is available")
        except ImportError:
            print("❌ Whisper not installed. Installing...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "openai-whisper"], check=True)
                print("✅ Whisper installed")
            except subprocess.CalledProcessError:
                print("❌ Failed to install Whisper")
    
    def setup_piper_tts(self):
        """Setup Piper TTS."""
        print("\n🔊 Setting up Piper TTS...")
        
        piper_dir = self.models_dir / "piper_voices"
        piper_dir.mkdir(exist_ok=True)
        
        # Check if Piper is installed
        if self.command_exists("piper"):
            print("✅ Piper TTS is already installed")
            self.download_piper_voices(piper_dir)
        else:
            print("⚠️  Piper TTS not found in PATH")
            print("Installing Piper TTS...")
            self.install_piper_tts(piper_dir)
    
    def install_piper_tts(self, piper_dir: Path):
        """Install Piper TTS."""
        try:
            # Try to install piper-tts Python package (may not work on all Python versions)
            result = subprocess.run([sys.executable, "-m", "pip", "install", "piper-tts"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Piper TTS Python package installed")
                # Download voices
                self.download_piper_voices(piper_dir)
            else:
                print("⚠️  Piper TTS Python package installation failed")
                print("This is normal - we'll use system TTS as fallback")
                print("For better TTS, install Piper manually from: https://github.com/rhasspy/piper")

        except Exception as e:
            print(f"⚠️  Piper TTS installation issue: {e}")
            print("System TTS will be used as fallback")
    
    def download_piper_voices(self, piper_dir: Path):
        """Download Piper voice models."""
        print("📥 Downloading Piper voice models...")
        
        # List of voice models to download
        voices = [
            {
                "name": "en_US-lessac-medium",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
                "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
            }
        ]
        
        for voice in voices:
            voice_file = piper_dir / f"{voice['name']}.onnx"
            config_file = piper_dir / f"{voice['name']}.onnx.json"
            
            if not voice_file.exists():
                print(f"📥 Downloading {voice['name']}...")
                try:
                    self.download_file(voice['url'], voice_file)
                    self.download_file(voice['config_url'], config_file)
                    print(f"✅ Downloaded {voice['name']}")
                except Exception as e:
                    print(f"❌ Failed to download {voice['name']}: {e}")
            else:
                print(f"✅ {voice['name']} already exists")
    
    def setup_ollama(self):
        """Setup Ollama for local LLM."""
        print("\n🧠 Setting up Ollama (optional)...")
        
        if self.command_exists("ollama"):
            print("✅ Ollama is already installed")
            
            # Check if model is available
            try:
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                if "llama3.1" in result.stdout:
                    print("✅ Llama 3.1 model is available")
                else:
                    print("📥 Downloading Llama 3.1 model (this may take a while)...")
                    subprocess.run(["ollama", "pull", "llama3.1:8b"], check=True)
                    print("✅ Llama 3.1 model downloaded")
            except subprocess.CalledProcessError:
                print("⚠️  Failed to setup Ollama model")
        else:
            print("⚠️  Ollama not found")
            print("To install Ollama:")
            print("  - Visit: https://ollama.ai")
            print("  - Or use: curl -fsSL https://ollama.ai/install.sh | sh")
            print("  - Then run: ollama pull llama3.1:8b")
    
    def create_directories(self):
        """Create necessary directories."""
        print("\n📁 Creating directories...")
        
        directories = [
            "data",
            "logs",
            "models/whisper",
            "models/piper_voices",
            "models/emotion_model"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {directory}")
    
    def download_file(self, url: str, filepath: Path):
        """Download a file with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as file, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
    
    def command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH."""
        try:
            subprocess.run([command, "--help"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def test_setup(self):
        """Test the setup by running basic functionality tests."""
        print("\n🧪 Testing setup...")
        
        # Test imports
        try:
            import whisper
            print("✅ Whisper import successful")
        except ImportError:
            print("❌ Whisper import failed")
        
        try:
            import pyaudio
            print("✅ PyAudio import successful")
        except ImportError:
            print("❌ PyAudio import failed")
        
        try:
            import librosa
            print("✅ Librosa import successful")
        except ImportError:
            print("❌ Librosa import failed")
        
        # Test audio devices
        try:
            import pyaudio
            audio = pyaudio.PyAudio()
            input_devices = audio.get_device_count()
            print(f"✅ Found {input_devices} audio devices")
            audio.terminate()
        except Exception as e:
            print(f"❌ Audio device test failed: {e}")
        
        print("🧪 Setup test complete")


def main():
    """Main setup function."""
    setup = ModelSetup()
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        setup.test_setup()
    else:
        setup.setup_all()
        setup.test_setup()


if __name__ == "__main__":
    main()
