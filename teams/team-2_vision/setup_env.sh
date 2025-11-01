#!/bin/bash

# ==========================
# Team Vulcans - Environment Setup
# ==========================

echo "==============================="
echo "  Setting up Python Environment"
echo "==============================="

# Check Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install it first."
    exit 1
fi

# Update packages (optional, safe to comment out)
sudo apt-get update -y

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists."
fi

# Activate environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Install all dependencies
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Optional: RPi-specific installs
if [ -f "/etc/rpi-issue" ]; then
    echo "🐧 Detected Raspberry Pi, installing Pi-specific libraries..."
    pip install picamera2 opencv-python-headless
fi

echo "✅ Environment setup complete!"
echo "To activate later: source venv/bin/activate"
