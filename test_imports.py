#!/usr/bin/env python3
"""
Test script to verify all imports are working correctly
"""

print("Testing imports for AI Detector Streamlit app...")
print("=" * 50)

try:
    import streamlit as st
    print("✅ Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

try:
    import torch
    print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    import torch.nn as nn
    print("✅ PyTorch nn module imported successfully")
except ImportError as e:
    print(f"❌ PyTorch nn import failed: {e}")

try:
    from torchvision import transforms
    print("✅ Torchvision transforms imported successfully")
except ImportError as e:
    print(f"❌ Torchvision import failed: {e}")

try:
    import numpy as np
    print(f"✅ NumPy imported successfully (version: {np.__version__})")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    from PIL import Image
    print("✅ PIL (Pillow) imported successfully")
except ImportError as e:
    print(f"❌ PIL import failed: {e}")

print("=" * 50)
print("Import test completed!")

# Test model creation
try:
    from app import Meso4
    model = Meso4()
    print("✅ Meso4 model created successfully")
    print(f"   Device: {model.device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"❌ Model creation failed: {e}")

print("=" * 50)
print("All tests completed!")
