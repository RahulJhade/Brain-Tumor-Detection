import streamlit as st
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
# Constants
IMG_SIZE = 150
class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']
# Page config
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠")
st.title("Brain Tumor Classifier 🧠")
st.write("Upload an MRI image to classify the type of brain tumor.")
@st.cache_resource
def get_model():
    return load_model('brain_tumor_model.keras')
try:
    model = get_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
# Function to predict and get full probabilities
def predict_image(image_bytes):
    # Convert uploaded file to opencv image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
    prediction = model.predict(img)[0]  # 1D array of probabilities
    top_class = np.argmax(prediction)
    return prediction, class_names[top_class], prediction[top_class]
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', width=300)
    st.write("Classifying...")
    
    # Get predictions
    bytes_data = uploaded_file.getvalue()
    probs, predicted_class, confidence = predict_image(bytes_data)
    st.success(f"Prediction: **{predicted_class.upper()}** ({confidence*100:.2f}%)")
    # Plot all class probabilities
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(class_names, probs, color=['orange', 'skyblue', 'lightgreen', 'tomato'])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Confidence")
    ax.set_title("Classification Probabilities")
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', fontsize=9)
        
    st.pyplot(fig)
