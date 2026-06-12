Brain Tumor Detection 🧠
A deep learning-based web application that classifies brain MRI scans into four categories:

Glioma
Meningioma
Pituitary Tumor
No Tumor
Live Demo: https://brain-tumor-detection-hfc4.onrender.com/

Features
Interactive web interface built with Streamlit
Fast and accurate predictions using a pre-trained TensorFlow/Keras model
Visual confidence breakdown with dynamic bar charts
How to Run Locally
Clone the repository:

bash

git clone https://github.com/RahulJhade/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
Install dependencies: Make sure you have Python installed, then run:

bash

pip install -r requirements.txt
(Note: If you are on Windows, ensure you have the Microsoft Visual C++ Redistributable installed for TensorFlow to work).

Start the application:

bash

streamlit run app.py
Tech Stack
Frontend/UI: Streamlit
Machine Learning: TensorFlow, Keras
Image Processing: OpenCV, Pillow
Data Visualization: Matplotlib
Deployment: Render
