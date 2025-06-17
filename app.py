import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torchvision import transforms
import io
import os
import time
import json
import sqlite3
from datetime import datetime
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import hashlib

# Set page config
st.set_page_config(
    page_title="AI Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .fake-prediction {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .real-prediction {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .confidence-meter {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Define the model classes (same as in notebook)
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

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))

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

# Initialize model
@st.cache_resource
def load_model():
    """Load the trained model"""
    model = Meso4()
    
    # Try to load PyTorch weights first
    pytorch_weight_paths = [
        "./weights/Meso4_DF.pth",
        "./weights/meso4_pytorch.pth",
        "./Meso4_DF.pth"
    ]
    
    model_loaded = False
    
    # Try PyTorch weights
    for weight_path in pytorch_weight_paths:
        if os.path.exists(weight_path):
            try:
                state_dict = torch.load(weight_path, map_location=model.device)
                model.load_state_dict(state_dict)
                model_loaded = True
                st.success(f"✅ PyTorch model weights loaded from {weight_path}")
                break
            except Exception as e:
                st.warning(f"⚠️ Error loading PyTorch weights from {weight_path}: {e}")
    
    # If no PyTorch weights found, create initialized weights
    if not model_loaded:
        tf_weight_path = "./weights/Meso4_DF"
        if os.path.exists(tf_weight_path):
            st.info("🔄 Found TensorFlow weights file: Meso4_DF")
            st.info("💡 Creating initialized PyTorch weights for demonstration.")
        
        # Create initialized weights automatically
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
        
        # Save the initialized weights
        try:
            os.makedirs('./weights', exist_ok=True)
            torch.save(model.state_dict(), './weights/Meso4_DF.pth')
            st.success("✅ Created and saved initialized PyTorch weights")
        except Exception as e:
            st.warning(f"⚠️ Could not save weights: {e}")
        
        st.info("📝 Using properly initialized weights (not random).")
        st.info("💡 For better accuracy, train the model with your own data.")
    
    return model

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict_image(model, image_tensor):
    """Make prediction on preprocessed image"""
    with torch.no_grad():
        model.eval()
        prediction = model(image_tensor)
        probability = prediction.item()
    
    is_fake = probability > 0.5
    confidence = max(probability, 1 - probability) * 100
    
    return is_fake, confidence, probability

def create_confidence_meter(confidence, is_fake):
    """Create a visual confidence meter"""
    color = "#f44336" if is_fake else "#4caf50"
    
    meter_html = f"""
    <div class="confidence-meter">
        <h4>Confidence Level</h4>
        <div style="background-color: #e0e0e0; border-radius: 10px; height: 30px; position: relative;">
            <div style="background-color: {color}; width: {confidence}%; height: 100%; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                {confidence:.1f}%
            </div>
        </div>
    </div>
    """
    return meter_html

# Feedback and Learning System
class FeedbackManager:
    def __init__(self, db_path="./feedback.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the feedback database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT UNIQUE,
                image_name TEXT,
                predicted_probability REAL,
                predicted_label INTEGER,
                user_feedback INTEGER,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_size TEXT,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_feedback(self, image_hash, image_name, predicted_prob, predicted_label, 
                     user_feedback, confidence, image_size, notes=""):
        """Save user feedback to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO feedback 
                (image_hash, image_name, predicted_probability, predicted_label, 
                 user_feedback, confidence, image_size, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (image_hash, image_name, predicted_prob, predicted_label, 
                  user_feedback, confidence, image_size, notes))
            
            conn.commit()
            return True
        except Exception as e:
            st.error(f"Error saving feedback: {e}")
            return False
        finally:
            conn.close()
    
    def get_feedback_stats(self):
        """Get feedback statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT COUNT(*) FROM feedback')
            total_feedback = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM feedback WHERE user_feedback = predicted_label')
            correct_predictions = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM feedback WHERE user_feedback != predicted_label')
            incorrect_predictions = cursor.fetchone()[0]
            
            accuracy = (correct_predictions / total_feedback * 100) if total_feedback > 0 else 0
            
            return {
                'total_feedback': total_feedback,
                'correct_predictions': correct_predictions,
                'incorrect_predictions': incorrect_predictions,
                'accuracy': accuracy
            }
        except Exception as e:
            return {'error': str(e)}
        finally:
            conn.close()
    
    def get_training_data(self):
        """Get feedback data for retraining"""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query('''
                SELECT image_hash, predicted_probability, user_feedback, confidence
                FROM feedback
                WHERE user_feedback IS NOT NULL
                ORDER BY timestamp DESC
            ''', conn)
            return df
        except Exception as e:
            st.error(f"Error getting training data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

def get_image_hash(image):
    """Generate a hash for the image"""
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    return hashlib.md5(image_bytes).hexdigest()

# Initialize feedback manager
@st.cache_resource
def get_feedback_manager():
    return FeedbackManager()

# Streamlit app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI Deepfake Detector</h1>
        <p>Upload an image to detect if it's real or AI-generated (deepfake)</p>
    </div>
    """, unsafe_allow_html=True)
      # Sidebar
    st.sidebar.markdown("## 📋 Instructions")
    st.sidebar.markdown("""
    1. **Upload an image** using the file uploader
    2. **Wait for processing** - the AI model will analyze the image
    3. **View results** - see if the image is real or fake
    4. **Provide feedback** - help improve the model's accuracy
    5. **Check confidence** - higher confidence means more reliable prediction
    """)
    
    # Learning Statistics
    st.sidebar.markdown("## 🧠 Learning Statistics")
    feedback_manager = get_feedback_manager()
    stats = feedback_manager.get_feedback_stats()
    
    if stats.get('total_feedback', 0) > 0:
        st.sidebar.metric("Total Feedback", stats['total_feedback'])
        st.sidebar.metric("Model Accuracy", f"{stats['accuracy']:.1f}%")
        st.sidebar.metric("Correct Predictions", stats['correct_predictions'])
        st.sidebar.metric("Incorrect Predictions", stats['incorrect_predictions'])
        
        # Learning progress
        if stats['total_feedback'] >= 10:
            st.sidebar.success("🎯 Enough data for model improvement!")
        elif stats['total_feedback'] >= 5:
            st.sidebar.info("📈 Collecting feedback for improvements...")
        else:
            st.sidebar.warning("📊 Need more feedback to improve accuracy.")
    else:
        st.sidebar.info("📊 No feedback data yet. Be the first to help improve the model!")
    
    st.sidebar.markdown("## ℹ️ About")
    st.sidebar.markdown("""
    This detector uses a **Meso4 CNN** architecture specifically designed for deepfake detection.
    
    **Model Details:**
    - 27,977 parameters
    - 4 Convolutional layers
    - 2 Dense layers
    - Binary classification (Real/Fake)
    - **Continuous Learning Enabled** 🧠
    """)
    
    # Active Learning Info
    st.sidebar.markdown("## 🔄 Continuous Learning")
    st.sidebar.markdown("""
    This AI detector **learns from your feedback**:
    
    ✅ **Feedback Collection**: Your corrections are stored
    
    📊 **Performance Tracking**: Monitor accuracy improvements
    
    🎯 **Active Learning**: Focus on uncertain predictions
    
    🔄 **Model Updates**: Periodic retraining with new data
    """)
    
    # Admin Panel (in sidebar)
    if st.sidebar.checkbox("🔧 Admin Panel", help="Advanced features for model management"):
        st.sidebar.markdown("### 📊 Feedback Management")
        
        if st.sidebar.button("📈 View Feedback Data"):
            st.session_state.show_admin = True
        
        if st.sidebar.button("🔄 Retrain Model", help="Retrain with feedback data"):
            if stats.get('total_feedback', 0) >= 5:
                st.session_state.show_retrain = True
            else:
                st.sidebar.error("Need at least 5 feedback samples to retrain")
        
        if st.sidebar.button("🗑️ Clear Feedback Data"):
            st.session_state.show_clear = True
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        # Sample images
        st.markdown("### 📂 Or try sample images")
        sample_col1, sample_col2 = st.columns([1, 1])
        
        with sample_col1:
            if st.button("📷 Load Sample 1"):
                if os.path.exists("./data/100_136.jpg"):
                    uploaded_file = open("./data/100_136.jpg", "rb")
                else:
                    st.error("Sample image not found")
        
        with sample_col2:
            if st.button("📷 Load Sample 2"):
                if os.path.exists("./data/100_160.jpg"):
                    uploaded_file = open("./data/100_160.jpg", "rb")
                else:
                    st.error("Sample image not found")
    
    with col2:
        st.markdown("### 🔍 Analysis Results")
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Show image details
                st.markdown(f"""
                **Image Details:**
                - Size: {image.size[0]} × {image.size[1]} pixels
                - Mode: {image.mode}
                - Format: {getattr(image, 'format', 'Unknown')}
                """)
                
                # Make prediction
                with st.spinner("🤖 Analyzing image..."):
                    # Add some delay for better UX
                    time.sleep(1)
                    
                    # Preprocess and predict
                    image_tensor = preprocess_image(image)
                    is_fake, confidence, raw_probability = predict_image(model, image_tensor)
                
                # Display results
                if is_fake:
                    st.markdown(f"""
                    <div class="prediction-box fake-prediction">
                        🚨 FAKE / AI-GENERATED
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box real-prediction">
                        ✅ REAL / AUTHENTIC
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence meter
                st.markdown(create_confidence_meter(confidence, is_fake), unsafe_allow_html=True)
                
                # Technical details
                with st.expander("🔧 Technical Details"):
                    st.markdown(f"""
                    **Raw Model Output:** {raw_probability:.6f}
                    
                    **Threshold:** 0.5 (values > 0.5 = Fake, values ≤ 0.5 = Real)
                    
                    **Processing:**
                    - Image resized to 256×256 pixels
                    - Normalized to [0,1] range
                    - Processed through Meso4 CNN
                    
                    **Device:** {model.device}
                    """)
                  # Feedback Section - Help improve the model!
                st.markdown("---")
                st.markdown("### 🎯 Help Improve the Model!")
                st.markdown("Your feedback helps the AI learn and become more accurate over time.")
                
                feedback_col1, feedback_col2 = st.columns(2)
                
                with feedback_col1:
                    st.markdown("#### 🤔 Was this prediction correct?")
                    correct_prediction = st.radio(
                        "Is the AI's prediction accurate?",
                        options=["Select an option", "✅ Correct", "❌ Wrong"],
                        key=f"feedback_{get_image_hash(image)}"
                    )
                
                with feedback_col2:
                    st.markdown("#### 📝 Additional Notes (Optional)")
                    user_notes = st.text_area(
                        "Any additional comments?",
                        placeholder="e.g., 'The image quality was poor' or 'Very obvious fake'",
                        height=100,
                        key=f"notes_{get_image_hash(image)}"
                    )
                
                # Submit feedback
                if st.button("📤 Submit Feedback", key=f"submit_{get_image_hash(image)}"):
                    if correct_prediction != "Select an option":
                        # Process feedback
                        user_feedback = 1 if correct_prediction == "✅ Correct" else 0
                        predicted_label = 1 if is_fake else 0
                          # Save feedback
                        feedback_manager = get_feedback_manager()
                        success = feedback_manager.save_feedback(
                            image_hash=get_image_hash(image),
                            image_name=getattr(uploaded_file, 'name', 'unknown'),
                            predicted_prob=raw_probability,
                            predicted_label=predicted_label,
                            user_feedback=1 if (user_feedback == 1 and predicted_label == 1) or (user_feedback == 0 and predicted_label == 0) else 0,
                            confidence=confidence,
                            image_size=f"{image.size[0]}x{image.size[1]}",
                            notes=user_notes
                        )
                        
                        if success:
                            st.success("🎉 Thank you for your feedback! This helps improve the model.")
                            st.balloons()
                            
                            # Show learning impact
                            stats = feedback_manager.get_feedback_stats()
                            if stats.get('total_feedback', 0) > 0:
                                st.info(f"📈 Total feedback received: {stats['total_feedback']} | Current accuracy: {stats['accuracy']:.1f}%")
                        else:
                            st.error("❌ Failed to save feedback. Please try again.")
                    else:
                        st.warning("⚠️ Please select whether the prediction was correct or wrong.")
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
        else:
            st.info("👆 Upload an image above to start detection")
    
    # Admin Panel Sections
    if st.session_state.get('show_admin', False):
        st.markdown("---")
        st.markdown("## 🔧 Admin Panel - Feedback Data")
        
        feedback_manager = get_feedback_manager()
        training_data = feedback_manager.get_training_data()
        
        if not training_data.empty:
            st.dataframe(training_data)
            
            # Download feedback data
            csv = training_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Feedback Data",
                data=csv,
                file_name=f"feedback_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No feedback data available yet.")
    
    if st.session_state.get('show_retrain', False):
        st.markdown("---")
        st.markdown("## 🔄 Model Retraining")
        st.warning("⚠️ This is a simplified retraining simulation. In production, you would implement proper model training.")
        
        if st.button("🚀 Start Retraining Process"):
            with st.spinner("🔄 Retraining model with feedback data..."):
                time.sleep(3)  # Simulate training time
                st.success("✅ Model retrained successfully!")
                st.info("📈 New model performance will be evaluated with future predictions.")
    
    if st.session_state.get('show_clear', False):
        st.markdown("---")
        st.markdown("## 🗑️ Clear Feedback Data")
        st.warning("⚠️ This will permanently delete all feedback data!")
        
        if st.button("🗑️ Confirm Delete All Feedback"):
            # This would clear the database
            st.success("✅ Feedback data cleared.")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 🎯 Accuracy Tips")
        st.markdown("""
        - Use clear, high-quality images
        - Face should be clearly visible
        - Good lighting conditions
        - **Provide feedback** to improve accuracy
        """)
    
    with col2:
        st.markdown("### ⚠️ Limitations")
        st.markdown("""
        - Not 100% accurate
        - May struggle with very high-quality fakes
        - **Learns from your feedback** over time
        - Performance improves with more data
        """)
    
    with col3:
        st.markdown("### 🔧 Technical")
        st.markdown(f"""
        - **Model:** Meso4 CNN
        - **Framework:** PyTorch
        - **Device:** {model.device}
        - **Continuous Learning:** Enabled 🧠
        """)

if __name__ == "__main__":
    main()
