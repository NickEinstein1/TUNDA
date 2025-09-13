#!/bin/bash

# Build script for Render deployment
echo "🚀 Starting Tunda build process..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements-render.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models/whisper
mkdir -p models/emotion_model
mkdir -p data
mkdir -p logs

# Setup models (lightweight for web deployment)
echo "🤖 Setting up models..."
python -c "
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Create a simple emotion classifier if it doesn't exist
model_path = 'models/emotion_classifier.pkl'
if not os.path.exists(model_path):
    print('Creating basic emotion classifier...')
    # Create a simple dummy classifier for deployment
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    # Train on dummy data
    X_dummy = np.random.rand(100, 10)
    y_dummy = np.random.choice(['happy', 'sad', 'angry', 'neutral'], 100)
    clf.fit(X_dummy, y_dummy)
    
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print('Basic emotion classifier created.')
else:
    print('Emotion classifier already exists.')
"

echo "✅ Build completed successfully!"
