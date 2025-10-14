#!/bin/bash

echo "--- Updating package list ---"
sudo apt-get update
sudo apt-get upgrade -y

echo "--- Installing essential dependencies for OpenCV ---"
sudo apt-get install -y build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran python3-dev

echo "--- Installing OpenCV for Python ---"
pip install opencv-python-headless --no-index --find-links=/usr/share/wheels/ --no-cache-dir

echo "--- Installing Picamera2 library ---"
pip install picamera2

echo "--- Installation Complete! ---"
