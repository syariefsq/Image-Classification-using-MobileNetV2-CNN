import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

# Load Model and Class Names
MODEL_PATH = './deployment/my_best_road_cnn_model.keras'
CLASS_NAMES_PATH = './deployment/class_names.pkl'
IMG_HEIGHT, IMG_WIDTH = 128, 128

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, 'rb') as f:
        return pickle.load(f)

model = load_model()
class_names = load_class_names()

# Preprocess Image
def preprocess_image(image, target_height, target_width):
    img = image.resize((target_width, target_height))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict
def predict_image_class(model, preprocessed_image, class_names):
    prob = model.predict(preprocessed_image)[0][0]
    idx = 1 if prob > 0.5 else 0
    return class_names[idx], prob

# Streamlit UI

def run():
    st.title("🛣️ Road Clean/Dirty Image Classifier")

    st.write("Upload a Road image to **Classify** as **Clean** or **Dirty**.")
    st.write("The model predicts whether the road is clean or dirty based on the uploaded image.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "gif", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        preprocessed = preprocess_image(image, IMG_HEIGHT, IMG_WIDTH)
        label, prob = predict_image_class(model, preprocessed, class_names)

        st.subheader(f"Prediction: {label}")
        st.write(f"Probability: {prob:.2%}")

        st.write("### Monitoring Insights:")

        if label.lower() == "dirty":
            st.info(
                "⚠️ **Dirty Road Detected!**\n\n"
                "This road is predicted as **Dirty**. "
                "Dirty roads can contribute to environmental pollution, health risks, and infrastructure damage. "
                "Consider prioritizing cleaning for this area."
            )
        else:
            st.success(
                "✅ **Clean Road Detected!**\n\n"
                "This road is predicted as **Clean**. "
                "Continue regular monitoring to maintain cleanliness."
            )

if __name__ == "__main__":
    run()