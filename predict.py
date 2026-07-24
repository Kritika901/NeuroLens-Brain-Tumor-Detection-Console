import json
import os

import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

MODEL_PATH = "models/brain_tumor_model.keras"
CLASS_INDICES_PATH = "models/class_indices.json"
IMAGE_SIZE = (224, 224)

_model = None
_class_names = None


def _load_model_and_classes():
    """Load the model and class labels once, then reuse them."""
    global _model, _class_names

    if _model is not None:
        return _model, _class_names

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at '{MODEL_PATH}'. Run 'python train.py' first."
        )

    if not os.path.exists(CLASS_INDICES_PATH):
        raise FileNotFoundError(
            f"'{CLASS_INDICES_PATH}' is missing. Run 'python train.py' first."
        )

    _model = load_model(MODEL_PATH)

    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)

    # class_indices.json maps class_name -> index, we need it the
    # other way around to turn a prediction index into a label.
    _class_names = {index: name for name, index in class_indices.items()}

    return _model, _class_names


def _normalize_class_key(raw_name):
    """
    Map a raw dataset folder name (e.g. 'glioma_tumor', 'no_tumor') to the
    short key used by the UI layer (e.g. 'glioma', 'notumor').
    """
    key = raw_name.strip().lower().replace("-", "_")
    key = key.replace("_tumor", "").replace("tumor_", "").replace("tumor", "")
    key = key.strip("_")
    if key in ("no", "notumor", ""):
        key = "notumor"
    return key


def _prepare_image(image_input):
    """
    Accepts either a file path (str) or an already-loaded PIL Image and
    returns a preprocessed, batched numpy array ready for the model.
    """
    if isinstance(image_input, (str, bytes, os.PathLike)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(
            "predict_image() expects a file path or a PIL.Image.Image instance, "
            f"got {type(image_input)!r}."
        )

    img = img.convert("RGB").resize(IMAGE_SIZE)
    img_array = np.asarray(img, dtype="float32")
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(image_input):
    """
    Predict the tumor class for a single MRI image.

    `image_input` can be either a file path or a PIL.Image.Image (e.g. the
    object returned by Image.open() in the Streamlit app).

    Returns a dict:
        {
            "class": "<short class key, e.g. 'glioma' / 'notumor'>",
            "confidence": <float, 0-1>,
            "probabilities": {"<short class key>": <float, 0-1>, ...},
        }
    """
    try:
        model, class_names = _load_model_and_classes()

        img_array = _prepare_image(image_input)

        prediction = model.predict(img_array, verbose=0)[0]
        predicted_index = int(np.argmax(prediction))

        keyed_probs = {
            _normalize_class_key(class_names[i]): float(prediction[i])
            for i in range(len(class_names))
        }
        predicted_class = _normalize_class_key(class_names[predicted_index])
        confidence = float(prediction[predicted_index])

        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": keyed_probs,
        }

    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not process this image: {exc}")