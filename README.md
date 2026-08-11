# Image Recognition System

This project is a simple image recognition application built with Streamlit and TensorFlow.
It uses MobileNetV2 for general image classification to predict object types such as animals and everyday objects.

## Files

- `app.py` - Streamlit frontend for uploading images and displaying classification predictions.
- `train_model.py` - Downloads the pretrained MobileNetV2 weights, saves the model locally, and stores it in a `.pkl` wrapper.
- `model_utils.py` - Contains helper functions for saving/loading a pickle-wrapped Keras model.

## Requirements

- Python 3.11
- TensorFlow
- Streamlit
- Pillow

Install dependencies with:

```bash
python -m pip install tensorflow streamlit pillow
```

## Usage

1. Train / download the model:

```bash
python train_model.py
```

This will create:
- `mobilenet_v2_model.h5`
- `mobilenet_v2_model.pkl`

2. Run the app:

```bash
streamlit run app.py
```

3. Upload an image and click `Predict`.

## Notes

- The current Streamlit app uses a pretrained MobileNetV2 model for object classification.
- The `.pkl` wrapper stores the model by saving/loading an H5 file during serialization, making it easier to persist the TensorFlow model in a pickle file.
