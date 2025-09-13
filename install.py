#!/usr/bin/env python3
"""
Safe installation script for Empathic Voice Companion.
Handles dependency compatibility issues gracefully.
"""

import sys
import subprocess
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro} detected")
    
    if version < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    elif version >= (3, 12):
        print("⚠️  Python 3.12+ detected - some packages may have compatibility issues")
        print("   We'll install compatible versions where possible")
    
    return True


def install_core_dependencies():
    """Install core dependencies that work across Python versions."""
    print("\n📦 Installing core dependencies...")
    
    core_packages = [
        "numpy>=1.21.0,<2.0.0",
        "scipy>=1.7.0,<2.0.0", 
        "scikit-learn>=1.0.0,<2.0.0",
        "joblib>=1.3.0,<2.0.0",
        "librosa>=0.9.2,<1.0.0",
        "soundfile>=0.10.3,<1.0.0",
        "openai-whisper>=20231117",
        "fastapi>=0.100.0,<1.0.0",
        "uvicorn>=0.23.0,<1.0.0",
        "websockets>=11.0.0,<12.0.0",
        "jinja2>=3.1.0,<4.0.0",
        "pyyaml>=6.0,<7.0.0",
        "requests>=2.28.0,<3.0.0",
        "click>=8.1.0,<9.0.0",
        "rich>=13.0.0,<14.0.0",
        "tqdm>=4.64.0,<5.0.0"
    ]
    
    for package in core_packages:
        try:
            print(f"Installing {package.split('>=')[0]}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {package.split('>=')[0]} installed successfully")
            else:
                print(f"⚠️  {package.split('>=')[0]} installation had issues:")
                print(f"   {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {package.split('>=')[0]} installation timed out")
        except Exception as e:
            print(f"❌ Failed to install {package.split('>=')[0]}: {e}")


def install_audio_dependencies():
    """Install audio dependencies with fallbacks."""
    print("\n🎵 Installing audio dependencies...")
    
    # PyAudio - often problematic
    try:
        print("Installing PyAudio...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "pyaudio"
        ], capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            print("✅ PyAudio installed successfully")
        else:
            print("⚠️  PyAudio installation failed")
            print("   This is common. Try one of these solutions:")
            if os.name == 'nt':  # Windows
                print("   - Install Visual Studio Build Tools")
                print("   - Or download PyAudio wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/")
            else:  # Linux/macOS
                print("   - sudo apt-get install portaudio19-dev (Ubuntu/Debian)")
                print("   - brew install portaudio (macOS)")
                
    except Exception as e:
        print(f"❌ PyAudio installation error: {e}")
    
    # pydub - usually works
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pydub>=0.25.1,<1.0.0"], 
                      check=True, capture_output=True, timeout=60)
        print("✅ pydub installed successfully")
    except Exception as e:
        print(f"⚠️  pydub installation failed: {e}")


def install_optional_dependencies():
    """Install optional dependencies that may have compatibility issues."""
    print("\n🔧 Installing optional dependencies...")
    
    optional_packages = [
        ("torch", "torch>=2.0.0", "PyTorch for advanced ML features"),
        ("transformers", "transformers>=4.21.0,<5.0.0", "Hugging Face transformers for LLM"),
        ("faster-whisper", "faster-whisper>=0.9.0", "Faster Whisper implementation"),
    ]
    
    for name, package, description in optional_packages:
        try:
            print(f"Installing {name} ({description})...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {name} installed successfully")
            else:
                print(f"⚠️  {name} installation failed - this is optional")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {name} installation timed out - skipping")
        except Exception as e:
            print(f"⚠️  {name} installation error: {e} - skipping")


def install_system_specific():
    """Install system-specific packages."""
    print("\n🖥️  Installing system-specific packages...")
    
    if os.name == 'nt':  # Windows
        try:
            print("Installing Windows-specific packages...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "pywin32>=306"
            ], capture_output=True, text=True, timeout=120)
            print("✅ Windows packages installed")
        except Exception as e:
            print(f"⚠️  Windows packages installation failed: {e}")


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "data",
        "logs", 
        "models",
        "models/whisper",
        "models/piper_voices",
        "models/emotion_model",
        "web",
        "web/templates",
        "web/static"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")


def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    # Test core imports
    test_imports = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("librosa", "Librosa"),
        ("whisper", "OpenAI Whisper"),
        ("fastapi", "FastAPI"),
        ("yaml", "PyYAML"),
    ]
    
    success_count = 0
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} import successful")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
    
    # Test optional imports
    optional_imports = [
        ("pyaudio", "PyAudio"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
    ]
    
    for module, name in optional_imports:
        try:
            __import__(module)
            print(f"✅ {name} (optional) import successful")
        except ImportError:
            print(f"⚠️  {name} (optional) not available")
    
    print(f"\n📊 Core imports: {success_count}/{len(test_imports)} successful")
    
    if success_count >= len(test_imports) - 1:  # Allow one failure
        print("🎉 Installation looks good!")
        return True
    else:
        print("⚠️  Some core dependencies failed to install")
        return False


def main():
    """Main installation function."""
    print("🚀 Empathic Voice Companion - Safe Installation")
    print("=" * 50)
    
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies in order of importance
    install_core_dependencies()
    install_audio_dependencies()
    install_optional_dependencies()
    install_system_specific()
    
    # Create directories
    create_directories()
    
    # Test installation
    success = test_installation()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Installation completed successfully!")
        print("\nNext steps:")
        print("1. Run: python test_system.py")
        print("2. If tests pass, run: python main.py")
        print("3. Or for web interface: python app.py")
        print("\nIf you encounter issues:")
        print("- Check the troubleshooting section in GETTING_STARTED.md")
        print("- Install missing dependencies manually")
        print("- The system will work with partial functionality")
    else:
        print("⚠️  Installation completed with some issues")
        print("The system may still work with reduced functionality")
        print("Check the error messages above and install missing packages manually")
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed with error: {e}")
        sys.exit(1)
