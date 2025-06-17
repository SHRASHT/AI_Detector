# 🔍 AI Deepfake Detector - Streamlit Web App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

A user-friendly web interface for detecting AI-generated (deepfake) images using the Meso4 CNN architecture with **continuous learning** capabilities.

## 🌟 New Feature: Continuous Learning System

This app now includes a **feedback learning system** that improves over time:
- 🎯 **User Feedback**: Mark predictions as correct/incorrect
- 📊 **Learning Statistics**: Track model accuracy in real-time
- 🧠 **Smart Improvements**: Model learns from your corrections
- 📈 **Performance Tracking**: Monitor accuracy trends over time

## 🚀 Quick Start

### 🌐 Online (Streamlit Cloud) - **Recommended**
**[Try the AI Detector Online →](https://your-app-url.streamlit.app)**

### 💻 Local Installation

### Method 1: Using the Batch File (Windows)
1. Double-click `run_app.bat`
2. Your browser will automatically open to `http://localhost:8501`

### Method 2: Using Command Line
1. Open terminal/command prompt
2. Navigate to the AI_Detector folder:
   ```bash
   cd "d:\Shreshth WebApps\AI_Detector"
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Open your browser and go to `http://localhost:8501`

## 📋 Prerequisites

Make sure you have installed all required packages:
```bash
pip install -r requirements.txt
```

## 🖼️ How to Use

1. **Upload Image**: Click "Browse files" and select an image (PNG, JPG, JPEG)
2. **Or Use Samples**: Try the built-in sample images
3. **View Results**: See if the image is classified as REAL or FAKE
4. **Check Confidence**: Higher confidence means more reliable prediction

## 🎯 Features

### ✅ **Core Functionality**
- **Real-time Detection**: Upload and analyze images instantly
- **Confidence Scoring**: See how confident the model is in its prediction
- **Sample Images**: Test with pre-loaded images
- **Responsive Design**: Works on desktop and mobile

### 🔧 **Technical Features**
- **Meso4 CNN Architecture**: Specialized for deepfake detection
- **PyTorch Backend**: Fast and efficient processing
- **Auto Device Detection**: Uses GPU if available, falls back to CPU
- **Image Preprocessing**: Automatic resizing and normalization

### 🎨 **User Interface**
- **Clean Design**: Modern, intuitive interface
- **Visual Feedback**: Color-coded results (Red=Fake, Green=Real)
- **Progress Indicators**: Loading animations during processing
- **Technical Details**: Expandable section with model information

## 📊 Model Information

- **Architecture**: Meso4 Convolutional Neural Network
- **Parameters**: 27,977 trainable parameters
- **Input Size**: 256×256×3 RGB images
- **Output**: Binary classification (Real vs Fake)
- **Framework**: PyTorch

### Model Architecture:
```
Input: 256×256×3 RGB Images
├── Conv2D(3→8) + BatchNorm + MaxPool(2×2)     → 128×128×8
├── Conv2D(8→8) + BatchNorm + MaxPool(2×2)     → 64×64×8  
├── Conv2D(8→16) + BatchNorm + MaxPool(2×2)    → 32×32×16
├── Conv2D(16→16) + BatchNorm + MaxPool(4×4)   → 8×8×16
├── Flatten                                     → 1024
├── Dense(1024→16) + LeakyReLU + Dropout(0.5)
└── Dense(16→1) + Sigmoid                       → [0,1] probability
```

## 📁 File Structure

```
AI_Detector/
├── app.py              # Main Streamlit application
├── run_app.bat         # Windows launcher script
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── data/              # Sample images
│   ├── 100_136.jpg
│   └── 100_160.jpg
├── weights/           # Model weights (optional)
│   └── Meso4_DF.pth
└── notebook/          # Development notebooks
    └── AI_detector.ipynb
```

## ⚙️ Configuration

### Model Weights
The app will automatically look for pre-trained weights in these locations:
1. `./weights/Meso4_DF.pth`
2. `../weights/Meso4_DF.pth`
3. `./Meso4_DF.pth`

If no weights are found, the app will use a randomly initialized model.

### Performance Tips
- **GPU Acceleration**: If you have CUDA installed, the app will automatically use GPU
- **Image Quality**: Use clear, high-quality images for best results
- **Face Visibility**: Ensure faces are clearly visible in the image

## 🔧 Troubleshooting

### Common Issues

1. **Module Not Found Error**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Port Already in Use**:
   ```bash
   streamlit run app.py --server.port 8502
   ```

3. **CUDA Errors**:
   - The app will automatically fall back to CPU if GPU is not available

### Error Messages
- **"Sample image not found"**: Make sure sample images are in the `data/` folder
- **"No pre-trained weights found"**: This is normal if you haven't trained the model yet

## 🎯 Accuracy Notes

- **Not 100% accurate**: Like all AI models, this detector has limitations
- **Training dependent**: Accuracy depends on the training data quality
- **High-quality fakes**: May struggle with very sophisticated deepfakes
- **Best practices**: Use multiple detection methods for critical applications

## 🚧 Future Enhancements

- [ ] Batch processing for multiple images
- [ ] Video frame analysis
- [ ] Model ensemble for improved accuracy
- [ ] Real-time webcam detection
- [ ] API endpoint for programmatic access

## 📧 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed
3. Verify Python version compatibility (3.8+)

## 🎉 Enjoy using your AI Deepfake Detector! 🚀
