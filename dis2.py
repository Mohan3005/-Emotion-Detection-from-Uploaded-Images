import streamlit as st
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define the directories for each emotion
EMOTION_DIRS = [
    "C:/Users/hp/Desktop/finalproject/train/angry",
    "C:/Users/hp/Desktop/finalproject/train/disgust",
    "C:/Users/hp/Desktop/finalproject/train/fear",
    "C:/Users/hp/Desktop/finalproject/train/happy",
    "C:/Users/hp/Desktop/finalproject/train/neutral",
    "C:/Users/hp/Desktop/finalproject/train/sad",
    "C:/Users/hp/Desktop/finalproject/train/surprise"
]

# Model path
MODEL_PATH =r"C:/Users/hp/Desktop/finalproject"

# Emotion labels corresponding to the directories
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the pre-trained model
@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None
    model = load_model(MODEL_PATH)
    return model

def main():
    st.title("Emotion Detection from Images")
    
    # Load the model
    model = load_emotion_model()
    if model is None:
        return
    
    # Find all image files in the specified directories
    image_files = []
    for emotion_dir in EMOTION_DIRS:
        if os.path.exists(emotion_dir):
            for file in os.listdir(emotion_dir):
                if file.lower().endswith(('jpg', 'jpeg', 'png')):
                    image_files.append(os.path.join(emotion_dir, file))
    
    # Check if there are images to display
    if not image_files:
        st.warning("No images found in the specified directories.")
        return
    
    # Select an image file from the list
    selected_image = st.selectbox("Select an image", image_files, format_func=lambda x: os.path.basename(x))
    
    if selected_image:
        # Display the selected image
        image = Image.open(selected_image)
        st.image(image, caption=os.path.basename(selected_image), use_column_width=True)
        
        # Preprocess the image for prediction
        preprocessed_image = preprocess_image(image)
        
        # Predict the emotion
        prediction = model.predict(preprocessed_image)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        
        # Display the predicted emotion
        st.success(f"Predicted Emotion: {predicted_emotion}")

def preprocess_image(image):
    """Preprocess the image to make it suitable for the model."""
    # Resize the image to 48x48 pixels as expected by the model
    image = image.resize((48, 48))
    # Convert to grayscale
    image = image.convert('L')
    # Convert to an array
    image = img_to_array(image)
    # Normalize
    image = image / 255.0
    # Expand dimensions to match model input shape (1, 48, 48, 1)
    image = np.expand_dims(image, axis=0)
    return image

if __name__ == "__main__":
    main()
