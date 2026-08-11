"""
app.py
-------
Streamlit frontend for the Image Recognition System.

This file ONLY contains the user interface. It is designed to be easily
connected to an existing backend that uses:
- TensorFlow + MobileNetV2 for Image Classification
- OpenCV Haar Cascade for Face Detection

Placeholder functions are used wherever the real backend logic should be
plugged in later. Simply replace the contents of those functions with your
actual model calls.
"""

import os
import numpy as np
import streamlit as st
from PIL import Image
# Attempt to import MobileNetV2 helpers from tensorflow.keras first,
# then fall back to standalone keras if necessary (some environments
# separate `keras` from `tensorflow` or have different packaging).
try:
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image
except Exception:
    try:
        from keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
        from keras.preprocessing import image as keras_image
    except Exception as e:
        raise ImportError(
            "Could not import MobileNetV2 utilities from `tensorflow.keras` or `keras`.\n"
            "Install TensorFlow (preferred) or Keras and ensure versions are compatible.\n\n"
            "Quick fix (in your active Python environment / virtualenv):\n"
            "  pip install --upgrade pip\n"
            "  pip install tensorflow\n\n"
            "If you specifically use standalone Keras, run:\n"
            "  pip install keras\n\n"
            "Notes: Use a Python version supported by the chosen package (for TensorFlow, Python 3.8-3.11).\n"
            "After installation restart Streamlit (Ctrl+C then `streamlit run app.py`)."
        ) from e

from model_utils import MODEL_PKL_PATH, load_model_from_pickle


# =========================================================
# BACKEND LOADING / INFERENCE FUNCTIONS
# =========================================================

def load_saved_model():
    """Load the saved model from the pickle wrapper."""
    if not os.path.exists(MODEL_PKL_PATH):
        raise FileNotFoundError(
            f"Pickle model not found. Run `python train_model.py` first to create {MODEL_PKL_PATH}."
        )
    return load_model_from_pickle(MODEL_PKL_PATH)


@st.cache_resource
def get_model():
    return load_saved_model()


def classify_image(image_obj):
    """Classify the uploaded image using MobileNetV2."""
    model = get_model()
    image_rgb = image_obj.convert("RGB")
    image_resized = image_rgb.resize((224, 224))
    image_array = keras_image.img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    preds = model.predict(image_array)
    decoded = decode_predictions(preds, top=3)[0]
    predictions = [
        {
            "label": item[1].replace("_", " "),
            "confidence": float(item[2] * 100),
        }
        for item in decoded
    ]
    return predictions


def run_prediction(image):
    """
    Runs image classification on the given image.
    Returns a dictionary with prediction results, or raises an exception on failure.
    """
    predictions = classify_image(image)
    return {"predictions": predictions}


# =========================================================
# STREAMLIT UI
# =========================================================

def render_sidebar():
    """Displays project information in the sidebar."""
    st.sidebar.header("Project Info")
    st.sidebar.write("**Project Name:** Image Recognition System")
    st.sidebar.write("**Model:** MobileNetV2")
    st.sidebar.write("**Task:** General Image Classification")
    st.sidebar.write("**Supported Formats:** JPG, JPEG, PNG")


def render_header():
    """Displays the title and short description."""
    st.title("Image Recognition System")
    st.write("Upload an image to predict the object type, such as cat, tiger, elephant, and more.")


def render_results(results):
    """Displays classification results."""
    st.subheader("Image Classification")
    top_prediction = results["predictions"][0]
    st.write(f"**Predicted Class:** {top_prediction['label']}")
    st.write(f"**Confidence Score:** {top_prediction['confidence']:.2f}%")

    if len(results["predictions"]) > 1:
        st.write("**Other top predictions:**")
        for prediction in results["predictions"][1:]:
            st.write(f"- {prediction['label']}: {prediction['confidence']:.2f}%")


def main():
    render_sidebar()
    render_header()

    st.divider()

    # Image uploader
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Please upload an image.")
        return

    # Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict button
    if st.button("Predict"):
        try:
            results = run_prediction(image)
            st.success("Prediction completed successfully!")
            st.divider()
            render_results(results)
        except Exception:
            st.error("Prediction failed. Please try again with a different image.")


if __name__ == "__main__":
    main()
