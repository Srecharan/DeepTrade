#!/usr/bin/env python3
"""
DeepTrade AI - Test Environment Setup Script
Sets up conda environment and installs dependencies for CPU testing
"""

import os
import sys
import subprocess
import json
import time

def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"Output: {result.stdout}")
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def check_conda_installed():
    """Check if conda is installed"""
    print("🔍 Checking conda installation...")
    result = subprocess.run("conda --version", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Conda found: {result.stdout.strip()}")
        return True
    else:
        print("❌ Conda not found. Please install Anaconda or Miniconda first.")
        print("Download from: https://docs.conda.io/en/latest/miniconda.html")
        return False

def create_conda_environment():
    """Create a new conda environment for testing"""
    env_name = "deeptrade-test"
    
    print(f"\n🆕 Creating conda environment: {env_name}")
    
    # Check if environment already exists
    result = subprocess.run(f"conda env list | grep {env_name}", 
                          shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Environment {env_name} already exists. Removing it first...")
        run_command(f"conda env remove -n {env_name} -y", "Removing existing environment")
    
    # Create new environment with Python 3.9
    success = run_command(
        f"conda create -n {env_name} python=3.9 -y",
        f"Creating {env_name} environment with Python 3.9"
    )
    
    return success, env_name

def install_basic_dependencies(env_name):
    """Install basic dependencies that work on CPU"""
    
    print(f"\n📦 Installing basic dependencies in {env_name}...")
    
    # Basic scientific computing packages
    basic_packages = [
        "numpy pandas scikit-learn matplotlib seaborn jupyter",
        "requests flask fastapi uvicorn",
        "click tqdm python-dotenv",
        "pytest pytest-cov"
    ]
    
    for packages in basic_packages:
        success = run_command(
            f"conda run -n {env_name} pip install {packages}",
            f"Installing: {packages}"
        )
        if not success:
            return False
    
    return True

def install_ml_dependencies_cpu(env_name):
    """Install ML dependencies for CPU usage"""
    
    print(f"\n🧠 Installing ML dependencies (CPU-only) in {env_name}...")
    
    # Install PyTorch CPU version
    success = run_command(
        f"conda run -n {env_name} pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "Installing PyTorch (CPU version)"
    )
    if not success:
        return False
    
    # Install transformers and other ML packages
    ml_packages = [
        "transformers datasets tokenizers",
        "xgboost lightgbm",
        "yfinance alpha-vantage"
    ]
    
    for packages in ml_packages:
        success = run_command(
            f"conda run -n {env_name} pip install {packages}",
            f"Installing: {packages}"
        )
        if not success:
            return False
    
    return True

def install_lightweight_streaming(env_name):
    """Install lightweight alternatives for streaming components"""
    
    print(f"\n🌊 Installing lightweight streaming components in {env_name}...")
    
    # Instead of full Kafka, use lightweight alternatives for testing
    streaming_packages = [
        "redis",  # For simple message queue testing
        "psycopg2-binary",  # For database connections
        "sqlalchemy",  # For database ORM
    ]
    
    for package in streaming_packages:
        success = run_command(
            f"conda run -n {env_name} pip install {package}",
            f"Installing: {package}"
        )
        if not success:
            return False
    
    return True

def test_installations(env_name):
    """Test that key packages are installed correctly"""
    
    print(f"\n🧪 Testing installations in {env_name}...")
    
    test_script = '''
import sys
print(f"Python version: {sys.version}")

# Test basic packages
try:
    import numpy as np
    import pandas as pd
    import sklearn
    print("✅ Basic packages: numpy, pandas, scikit-learn")
except ImportError as e:
    print(f"❌ Basic packages error: {e}")

# Test PyTorch (CPU)
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
except ImportError as e:
    print(f"❌ PyTorch error: {e}")

# Test transformers
try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers error: {e}")

# Test other ML packages
try:
    import xgboost as xgb
    print(f"✅ XGBoost: {xgb.__version__}")
except ImportError as e:
    print(f"❌ XGBoost error: {e}")

# Test API packages
try:
    import yfinance as yf
    import requests
    print("✅ API packages: yfinance, requests")
except ImportError as e:
    print(f"❌ API packages error: {e}")

print("\\n🎉 Installation test completed!")
'''
    
    # Write test script to temporary file
    with open('temp_test.py', 'w') as f:
        f.write(test_script)
    
    success = run_command(
        f"conda run -n {env_name} python temp_test.py",
        "Running installation test"
    )
    
    # Clean up
    if os.path.exists('temp_test.py'):
        os.remove('temp_test.py')
    
    return success

def create_test_config():
    """Create configuration for CPU testing"""
    
    print("\n⚙️ Creating test configuration...")
    
    test_config = {
        "testing_mode": True,
        "device": "cpu",
        "distributed_training": {
            "world_size": 1,  # Single process for CPU
            "backend": "gloo",  # CPU-compatible backend
            "simulate_multi_gpu": True
        },
        "kafka_streaming": {
            "use_redis_alternative": True,
            "simulate_kafka": True,
            "lightweight_mode": True
        },
        "airflow": {
            "use_sqlite": True,
            "local_executor": True,
            "lightweight_mode": True
        },
        "api_keys": {
            "news_api": "demo_key",
            "reddit_client_id": "demo_client",
            "reddit_client_secret": "demo_secret",
            "sec_api": "demo_key"
        }
    }
    
    os.makedirs('config', exist_ok=True)
    with open('config/test_config.json', 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print("✅ Test configuration saved to config/test_config.json")
    return True

def main():
    """Main setup function"""
    
    print("🚀 DeepTrade AI - Test Environment Setup")
    print("=" * 50)
    
    # Check conda
    if not check_conda_installed():
        return False
    
    # Create environment
    success, env_name = create_conda_environment()
    if not success:
        print("❌ Failed to create conda environment")
        return False
    
    # Install dependencies
    steps = [
        (install_basic_dependencies, "basic dependencies"),
        (install_ml_dependencies_cpu, "ML dependencies (CPU)"),
        (install_lightweight_streaming, "lightweight streaming components"),
    ]
    
    for install_func, description in steps:
        print(f"\n📦 Installing {description}...")
        if not install_func(env_name):
            print(f"❌ Failed to install {description}")
            return False
        print(f"✅ Successfully installed {description}")
    
    # Test installations
    if not test_installations(env_name):
        print("❌ Installation tests failed")
        return False
    
    # Create test config
    create_test_config()
    
    print(f"\n🎉 Environment setup completed successfully!")
    print(f"\nTo activate the environment:")
    print(f"conda activate {env_name}")
    print(f"\nTo test the infrastructure:")
    print(f"conda run -n {env_name} python scripts/test_infrastructure_functional.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 