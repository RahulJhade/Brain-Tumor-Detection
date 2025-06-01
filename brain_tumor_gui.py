import tkinter as tk
from tkinter import filedialog, Label, Frame
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Constants
IMG_SIZE = 150
class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']

# Load model
model = load_model('brain_tumor_model.keras')

# Function to predict and get full probabilities
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(img)[0]  # 1D array of probabilities
    top_class = np.argmax(prediction)
    return prediction, class_names[top_class], prediction[top_class]

# Function to display prediction
def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and display image
        image = Image.open(file_path).resize((200, 200))
        img_display = ImageTk.PhotoImage(image)
        image_panel.config(image=img_display)
        image_panel.image = img_display

        # Get predictions
        probs, predicted_class, confidence = predict_image(file_path)

        # Display textual result
        result_label.config(text=f"Prediction: {predicted_class.upper()} ({confidence*100:.2f}%)")

        # Plot all class probabilities
        plt.figure(figsize=(4, 3))
        bars = plt.bar(class_names, probs, color=['orange', 'skyblue', 'lightgreen', 'tomato'])
        plt.ylim([0, 1])
        plt.ylabel("Confidence")
        plt.title("Classification Probabilities")
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig("prediction_plot.png")
        plt.close()

        # Display chart
        chart_img = Image.open("prediction_plot.png")
        chart_img = chart_img.resize((320, 240))
        chart_tk = ImageTk.PhotoImage(chart_img)
        chart_panel.config(image=chart_tk)
        chart_panel.image = chart_tk

# Create main window
root = tk.Tk()
root.title("Brain Tumor Classifier")
root.geometry("500x650")
root.resizable(False, False)
root.configure(bg="#f0f4f8")

# Top frame for image
image_frame = Frame(root, bg="#f0f4f8")
image_frame.pack(pady=10)
image_panel = Label(image_frame, bg="#f0f4f8")
image_panel.pack()

# Button
upload_btn = tk.Button(root, text="Upload MRI Image", command=upload_and_predict, bg="#4caf50", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5)
upload_btn.pack(pady=10)

# Result
result_label = Label(root, text="Prediction: ", font=("Arial", 14), bg="#f0f4f8")
result_label.pack(pady=10)

# Chart display
chart_panel = Label(root, bg="#f0f4f8")
chart_panel.pack(pady=10)

root.mainloop()
