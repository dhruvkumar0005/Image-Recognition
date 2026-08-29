"""
train_model.py
----------------
This script does NOT train a custom model.

MobileNetV2 is already pretrained on the ImageNet dataset, so this script
simply downloads the pretrained weights ONE TIME and saves the model
locally. This way, the Streamlit app can load the model quickly from disk
instead of re-downloading the weights on every run.

Run this file once before starting the Streamlit app:
    python train_model.py
"""

import importlib
import os
import subprocess
import sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    tf = importlib.import_module("tensorflow")
except Exception as exc:
    raise ImportError("TensorFlow is required to run this project.") from exc

try:
    MobileNetV2 = importlib.import_module("tensorflow.keras.applications").MobileNetV2
except Exception:
    try:
        MobileNetV2 = importlib.import_module("keras.applications").MobileNetV2
    except Exception as exc:
        raise ImportError(
            "Could not import MobileNetV2 from TensorFlow or standalone Keras."
        ) from exc

from model_utils import MODEL_H5_PATH, MODEL_PKL_PATH, save_model_as_pickle

# Local path where the pretrained model will be saved in H5 format
MODEL_SAVE_PATH = MODEL_H5_PATH


def show_tensorflow_version():
    """Print the installed TensorFlow version."""
    print(f"TensorFlow Version: {tf.__version__}")


def load_pretrained_model():
    """
    Load the MobileNetV2 model with weights pretrained on ImageNet.
    Returns the loaded Keras model.
    """
    print("Loading pretrained MobileNetV2 model (ImageNet weights)...")
    model = MobileNetV2(weights="imagenet")
    print("Model loaded successfully!")
    return model


def save_model_locally(model, save_path=MODEL_SAVE_PATH):
    """Save the model to disk so it can be reloaded quickly later."""
    print(f"Saving model to '{save_path}'...")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path, include_optimizer=False)
    print("Model saved successfully!")


def save_model_pickle(model, pickle_path=MODEL_PKL_PATH, h5_path=MODEL_H5_PATH):
    """Save the Keras model wrapper as a pickle file."""
    print(f"Saving pickle wrapper to '{pickle_path}'...")
    save_model_as_pickle(model, pickle_path=pickle_path, h5_path=h5_path)
    print("Pickle file saved successfully!")


def launch_app():
    """Launch the Streamlit app after the model has been saved."""
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    if not os.path.exists(app_path):
        print(f"Could not find app.py at {app_path}. Skipping launch.")
        return

    print("Launching Streamlit app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=False)


def main():
    show_tensorflow_version()
    model = load_pretrained_model()
    save_model_locally(model)
    save_model_pickle(model)

    print("\nMobileNetV2 Model Downloaded, Saved, and Pickled Successfully!")
    print(f"H5 model saved at: {MODEL_SAVE_PATH}")
    print(f"Pickle file saved at: {MODEL_PKL_PATH}")

    launch_app()


if __name__ == "__main__":
    main()
