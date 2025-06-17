#!/usr/bin/env python3
"""
Convert TensorFlow/Keras weights to PyTorch format
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add current directory to path to import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define the PyTorch model (same as in app.py)
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32).to(self.device)
            return self.forward(x)

class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        super(Meso4, self).__init__()
        
        # Define the model architecture
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(4, 4)
        
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.dropout1 = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Move to device
        self.to(self.device)
        
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout1(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc2(x))
        
        return x

def convert_tensorflow_to_pytorch():
    """
    Convert TensorFlow weights to PyTorch format
    """
    print("🔄 Starting TensorFlow to PyTorch weight conversion...")
    
    # Check if TensorFlow weights exist
    tf_weights_path = "./weights/Meso4_DF"
    if not os.path.exists(tf_weights_path):
        print(f"❌ TensorFlow weights not found at {tf_weights_path}")
        return False
    
    try:
        # Try to load TensorFlow/Keras
        print("📦 Attempting to import TensorFlow...")
        import tensorflow as tf
        print(f"✅ TensorFlow imported successfully (version: {tf.__version__})")
        
        # Load the TensorFlow model
        print("🔄 Loading TensorFlow model...")
        tf_model = tf.keras.models.load_model(tf_weights_path)
        print("✅ TensorFlow model loaded successfully")
        
        # Print model summary
        print("\n📋 TensorFlow Model Summary:")
        tf_model.summary()
        
        # Create PyTorch model
        print("\n🔄 Creating PyTorch model...")
        pytorch_model = Meso4()
        print("✅ PyTorch model created successfully")
        
        # Convert weights layer by layer
        print("\n🔄 Converting weights...")
        
        # Get TensorFlow weights
        tf_weights = tf_model.get_weights()
        print(f"📊 Found {len(tf_weights)} weight tensors in TensorFlow model")
        
        # Map TensorFlow layers to PyTorch layers
        pytorch_state_dict = pytorch_model.state_dict()
        
        # This is a manual mapping - you may need to adjust based on the actual model structure
        layer_mapping = [
            # Conv1 layer
            ('conv1.weight', 0),  # Conv2D weights
            ('conv1.bias', 1),    # Conv2D bias
            ('bn1.weight', 2),    # BatchNorm gamma
            ('bn1.bias', 3),      # BatchNorm beta
            ('bn1.running_mean', 4),  # BatchNorm moving mean
            ('bn1.running_var', 5),   # BatchNorm moving variance
            
            # Conv2 layer
            ('conv2.weight', 6),
            ('conv2.bias', 7),
            ('bn2.weight', 8),
            ('bn2.bias', 9),
            ('bn2.running_mean', 10),
            ('bn2.running_var', 11),
            
            # Conv3 layer
            ('conv3.weight', 12),
            ('conv3.bias', 13),
            ('bn3.weight', 14),
            ('bn3.bias', 15),
            ('bn3.running_mean', 16),
            ('bn3.running_var', 17),
            
            # Conv4 layer
            ('conv4.weight', 18),
            ('conv4.bias', 19),
            ('bn4.weight', 20),
            ('bn4.bias', 21),
            ('bn4.running_mean', 22),
            ('bn4.running_var', 23),
            
            # Dense layers
            ('fc1.weight', 24),
            ('fc1.bias', 25),
            ('fc2.weight', 26),
            ('fc2.bias', 27),
        ]
        
        # Convert weights
        converted_weights = {}
        for pytorch_name, tf_index in layer_mapping:
            if tf_index < len(tf_weights):
                tf_weight = tf_weights[tf_index]
                
                # Convert TensorFlow weight to PyTorch format
                if 'conv' in pytorch_name and 'weight' in pytorch_name:
                    # TensorFlow conv weights: [height, width, in_channels, out_channels]
                    # PyTorch conv weights: [out_channels, in_channels, height, width]
                    pytorch_weight = np.transpose(tf_weight, (3, 2, 0, 1))
                elif 'fc' in pytorch_name and 'weight' in pytorch_name:
                    # TensorFlow dense weights: [in_features, out_features]
                    # PyTorch linear weights: [out_features, in_features]
                    pytorch_weight = np.transpose(tf_weight)
                else:
                    # Bias and BatchNorm parameters don't need transposition
                    pytorch_weight = tf_weight
                
                # Convert to PyTorch tensor
                converted_weights[pytorch_name] = torch.from_numpy(pytorch_weight.astype(np.float32))
                print(f"✅ Converted {pytorch_name}: {tf_weight.shape} -> {converted_weights[pytorch_name].shape}")
            else:
                print(f"⚠️ Warning: {pytorch_name} not found in TensorFlow weights")
        
        # Load converted weights into PyTorch model
        print("\n🔄 Loading converted weights into PyTorch model...")
        pytorch_model.load_state_dict(converted_weights, strict=False)
        
        # Save PyTorch weights
        pytorch_weights_path = "./weights/Meso4_DF.pth"
        torch.save(pytorch_model.state_dict(), pytorch_weights_path)
        print(f"✅ PyTorch weights saved to {pytorch_weights_path}")
        
        print("\n🎉 Weight conversion completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ TensorFlow not available: {e}")
        print("💡 To convert weights, install TensorFlow: pip install tensorflow")
        return False
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

def create_dummy_weights():
    """
    Create dummy trained weights for demonstration
    """
    print("🔄 Creating dummy trained weights...")
    
    # Create model
    model = Meso4()
    
    # Initialize with some reasonable values (not completely random)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'conv' in name:
                    # Xavier initialization for conv layers
                    nn.init.xavier_uniform_(param)
                elif 'fc' in name:
                    # Xavier initialization for linear layers
                    nn.init.xavier_uniform_(param)
                elif 'bn' in name:
                    # BatchNorm weights
                    nn.init.ones_(param)
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.zeros_(param)
    
    # Save the initialized weights
    dummy_weights_path = "./weights/Meso4_DF_dummy.pth"
    torch.save(model.state_dict(), dummy_weights_path)
    print(f"✅ Dummy weights saved to {dummy_weights_path}")
    
    return True

if __name__ == "__main__":
    print("🚀 AI Detector Weight Conversion Tool")
    print("=" * 50)
    
    # First try to convert TensorFlow weights
    if convert_tensorflow_to_pytorch():
        print("\n✅ TensorFlow weights converted successfully!")
    else:
        print("\n⚠️ TensorFlow conversion failed. Creating dummy weights instead...")
        create_dummy_weights()
        print("\n💡 For better accuracy, either:")
        print("   1. Install TensorFlow and run this script again")
        print("   2. Train the model with your own data")
        print("   3. Use the PyTorch notebook to train the model")
    
    print("\n🎯 Next steps:")
    print("   - Run the Streamlit app: python -m streamlit run app.py")
    print("   - The app will now use the converted/dummy weights")
