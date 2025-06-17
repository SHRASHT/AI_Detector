#!/usr/bin/env python3
"""
Simple script to create PyTorch weights for the AI detector
"""

import torch
import torch.nn as nn
import os

# Define the PyTorch model (same as in app.py)
class Meso4(nn.Module):
    def __init__(self):
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

def create_initialized_weights():
    """Create properly initialized weights for the model"""
    print("Creating initialized PyTorch weights...")
    
    # Create model
    model = Meso4()
    
    # Initialize weights with Xavier/He initialization
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    # Apply initialization
    model.apply(init_weights)
    
    # Save the weights
    os.makedirs('./weights', exist_ok=True)
    weight_path = './weights/Meso4_DF.pth'
    torch.save(model.state_dict(), weight_path)
    
    print(f"✅ Initialized weights saved to {weight_path}")
    print(f"📊 Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    return True

if __name__ == "__main__":
    print("🚀 Creating PyTorch weights for AI Detector")
    print("=" * 50)
    
    try:
        create_initialized_weights()
        print("\n✅ PyTorch weights created successfully!")
        print("\n💡 The app will now use these initialized weights.")
        print("   For better accuracy, consider training the model with your data.")
    except Exception as e:
        print(f"❌ Error creating weights: {e}")
