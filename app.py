import streamlit as st
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Define class names
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# --- App Configuration ---
st.set_page_config(
    page_title="Flower Classification App",
    page_icon="🌸",
    layout="centered"
)

# --- Model Loading ---
@st.cache_resource
def load_my_model():
    """Loads the pre-trained Keras model from disk."""
    return load_model('flower_cnn_model.h5')

model = load_my_model()

# --- Prediction Function ---
def predict_flower(image: Image.Image, model):
    """Preprocesses the image and returns prediction."""
    img = image.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

# --- Main App Interface ---
st.title("Flower Classification App")
st.markdown(
    "Upload a flower image or select an example to let the deep learning model predict its species. "
    "The model can identify **daisies, dandelions, roses, sunflowers, and tulips**."
)

# --- Sidebar ---
st.sidebar.title("About the App")
st.sidebar.info(
    "This application is a demonstration of a Convolutional Neural Network (CNN) "
    "trained to classify images of flowers. Upload an image and the model will "
    "predict the flower type."
)

st.sidebar.header("Upload Your Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

st.sidebar.header("Or Try an Example")
example_path = "example_images"
if st.sidebar.button("Daisy Example"):
    uploaded_file = os.path.join(example_path, "daisy.jpg")
if st.sidebar.button("Rose Example"):
    uploaded_file = os.path.join(example_path, "rose.jpg")
if st.sidebar.button("Tulip Example"):
    uploaded_file = os.path.join(example_path, "tulip.jpg")

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Base, Angelo using Streamlit Cloud for Final Exam")


if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            with st.spinner('Analyzing image...'):
                prediction = predict_flower(image, model)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction)

            st.subheader("Prediction")
            st.success(f"This looks like a **{predicted_class}**.")
            st.metric(label="Confidence", value=f"{confidence:.2%}")

            st.subheader("All Probabilities")
            probs_df = pd.DataFrame(prediction[0], index=class_names, columns=["Probability"])
            probs_df["Probability"] = probs_df["Probability"].apply(lambda x: f"{x:.2%}")
            st.dataframe(probs_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Could not process the image. Please ensure it's a valid image file and try again.")
else:
    st.info("Awaiting an image to classify...")