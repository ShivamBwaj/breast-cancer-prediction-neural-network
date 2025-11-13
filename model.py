import os
import joblib
import numpy as np

"""Model loader that supports Keras (.h5) or scikit-learn (joblib) models.

Usage:
  model_tuple = load_model()
  proba, pred = predict(model_tuple, X)

This file keeps TensorFlow import optional so the app can still run with only
scikit-learn present.
"""

try:
    from tensorflow.keras.models import load_model as keras_load_model
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

MODEL_DIR = os.path.dirname(__file__)
MODEL_H5 = os.path.join(MODEL_DIR, 'model.h5')
MODEL_JOBLIB = os.path.join(MODEL_DIR, 'model.joblib')
SCALER_JOBLIB = os.path.join(MODEL_DIR, 'scaler.joblib')


def load_model(path_h5=MODEL_H5, path_joblib=MODEL_JOBLIB, scaler_path=SCALER_JOBLIB):
    """Load a model and optional scaler. Returns (kind, model, scaler).

    kind: 'keras' or 'sklearn'
    model: loaded model object
    scaler: sklearn transformer or None
    """
    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None

    # Prefer Keras model if available
    if KERAS_AVAILABLE and os.path.exists(path_h5):
        model = keras_load_model(path_h5)
        return ('keras', model, scaler)

    if os.path.exists(path_joblib):
        model = joblib.load(path_joblib)
        return ('sklearn', model, scaler)

    raise FileNotFoundError(
        f"No model found. Place a Keras model at '{path_h5}' or a sklearn joblib at '{path_joblib}'."
    )


def predict(model_tuple, X):
    """Return (proba, pred) for input X shaped (1, n_features).

    model_tuple is (kind, model, scaler).
    """
    kind, model, scaler = model_tuple
    X = np.asarray(X)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass

    if kind == 'keras':
        preds = model.predict(X)
        # handle common shapes: (1,1) sigmoid or (1,2) softmax
        if hasattr(preds, 'ndim') and preds.ndim == 2 and preds.shape[1] == 1:
            proba = float(preds[0, 0])
        else:
            proba = float(preds[0, 1])
        pred = int(proba >= 0.5)
        return proba, pred

    # sklearn fallback
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    return proba, pred
