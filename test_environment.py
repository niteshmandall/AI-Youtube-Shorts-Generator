import sys
import subprocess
import os
import importlib.metadata
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.10 or higher"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    if version < (3, 10):
        print("❌ Python 3.10 or higher is required")
        return False
    print("✅ Python version is compatible")
    return True

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        version_line = result.stdout.split('\n')[0]
        print(f"🎥 {version_line}")
        print("✅ FFmpeg is installed")
        return True
    except FileNotFoundError:
        print("❌ FFmpeg is not installed or not in PATH")
        return False

def check_imagemagick():
    """Check if ImageMagick is installed and has correct policy"""
    try:
        result = subprocess.run(['convert', '--version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        version_line = result.stdout.split('\n')[0]
        print(f"🖼️  {version_line}")
        
        # Test basic text rendering
        test_cmd = [
            'convert',
            '-background', 'none',
            '-fill', 'white',
            '-font', 'Arial',
            '-pointsize', '24',
            '-size', '320x',
            'label:Test Caption',
            'test_caption.png'
        ]
        subprocess.run(test_cmd, check=True)
        
        if os.path.exists('test_caption.png'):
            os.remove('test_caption.png')
            print("✅ ImageMagick can render text")
            return True
        else:
            print("❌ ImageMagick test render failed")
            return False
            
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"❌ ImageMagick test failed: {str(e)}")
        return False

def check_python_packages():
    """Check if all required Python packages are installed"""
    required = {
        'moviepy', 'faster-whisper', 'pydub', 'numpy', 'opencv-python',
        'google-generativeai', 'python-dotenv', 'imageio', 'imageio-ffmpeg'
    }
    
    missing = set()
    installed = set()
    
    for pkg in required:
        try:
            version = importlib.metadata.version(pkg)
            installed.add(pkg)
            print(f"✅ {pkg:20} v{version}")
        except importlib.metadata.PackageNotFoundError:
            missing.add(pkg)
    
    if missing:
        print("\n❌ Missing packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing))
    
    return len(missing) == 0

def check_directories():
    """Check if required directories exist"""
    required_dirs = ['videos', 'Components']
    all_exist = True
    
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists() and path.is_dir():
            print(f"📂 {dir_name:15} exists")
        else:
            print(f"❌ {dir_name:15} missing")
            all_exist = False
    
    return all_exist

def check_environment():
    """Run all environment checks"""
    print("🚀 Running environment checks...\n")
    
    results = [
        ("Python Version", check_python_version()),
        ("FFmpeg", check_ffmpeg()),
        ("ImageMagick", check_imagemagick()),
        ("Python Packages", check_python_packages()),
        ("Project Structure", check_directories())
    ]
    
    print("\n📊 Results:")
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{name:20} [{'✅' if success else '❌'}] {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All checks passed! You're ready to run the application.")
    else:
        print("\n❌ Some checks failed. Please address the issues above before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    check_environment()
