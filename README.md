# Breast Cancer Predictor (Flask)

This is a minimal Flask web app that loads a model and serves a form to input 30 features (matching sklearn's breast cancer dataset) and returns a prediction.

The app supports two model types:
- Keras/TensorFlow models saved as `model.h5` (neural network, recommended if you used the notebook)
- scikit-learn models saved with joblib as `model.joblib`

Quick start (Windows PowerShell):

1. Create a virtual environment and activate:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train example models:

- Sklearn example (creates `model.joblib`):

```powershell
python train_dummy_model.py
```

- Neural-net example (creates `model.h5` and `scaler.joblib`):

```powershell
python train_dummy_nn.py
```

4. Run the app:

```powershell
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

Replacing the model:
- For Keras: replace `model.h5` with your Keras model. The app will auto-detect it if TensorFlow is installed.
- For sklearn: replace `model.joblib` with your trained scikit-learn model that implements `predict` and `predict_proba`.

Security note:
- This sample is for local development and demonstration only.
