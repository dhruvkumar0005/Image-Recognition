import importlib
import os
import pickle

try:
    tf = importlib.import_module("tensorflow")
except Exception as exc:
    raise ImportError("TensorFlow is required to run this project.") from exc

MODEL_H5_PATH = "mobilenet_v2_model.h5"
MODEL_PKL_PATH = "mobilenet_v2_model.pkl"


class PickleableKerasModel:
    """Lightweight wrapper to make a Keras model pickleable.

    The model is saved to an H5 file during pickle serialization, and
    reloaded from that file during deserialization.
    """

    def __init__(self, model, h5_path=MODEL_H5_PATH):
        self._model = model
        self.h5_path = h5_path

    def get_model(self):
        if self._model is None:
            self._model = tf.keras.models.load_model(self.h5_path)
        return self._model

    def __getstate__(self):
        state = {"h5_path": self.h5_path}
        if self._model is not None:
            os.makedirs(os.path.dirname(self.h5_path) or ".", exist_ok=True)
            self._model.save(self.h5_path, include_optimizer=False)
        return state

    def __setstate__(self, state):
        self.h5_path = state["h5_path"]
        self._model = tf.keras.models.load_model(self.h5_path)


def save_model_as_pickle(model, pickle_path=MODEL_PKL_PATH, h5_path=MODEL_H5_PATH):
    wrapper = PickleableKerasModel(model, h5_path=h5_path)
    with open(pickle_path, "wb") as file_obj:
        pickle.dump(wrapper, file_obj)
    return pickle_path


def load_model_from_pickle(pickle_path=MODEL_PKL_PATH):
    with open(pickle_path, "rb") as file_obj:
        wrapper = pickle.load(file_obj)
    return wrapper.get_model()
