from ctypes import alignment
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('./model/model.h5')

def preprocess_image(img):
    img = img.resize((150, 150))  # Resize the image
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

# Streamlit app
st.title("Rock-Paper-Scissors Prediction")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file).convert('RGB')
    img_array = preprocess_image(img)

    # Predict the label
    predictions = model.predict(img_array)
    labels = ['Paper', 'Rock', 'Scissors']
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = labels[predicted_index]

    # Display the image and prediction
    st.image(img, use_column_width=True)
    st.markdown(f"<h3 style='text-align: center;'>Predicted Label: {predicted_label}</h3>", unsafe_allow_html=True)
    st.write("Confidence result: ")
    for i, j in enumerate(predictions[0]):
        st.write(f"{labels[i]} : {j:.4f}")