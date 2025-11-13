# Breast Cancer Predictor (Streamlit)

This is a Streamlit web app that loads a trained model and provides an interactive interface to:
- Randomize data (pick a random sample from the breast cancer dataset)
- Edit 30 features manually
- Get instant predictions with probability and risk assessment

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
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

Features:
- **🎲 Randomize Data**: Click to load a random sample from the dataset
- **🔮 Predict**: Click to get a prediction with probability
- **Manual input**: Edit any feature value with number inputs
- **Risk visualization**: Benign (🟢) or Malignant (🔴) with confidence scores

Replacing the model:
- For Keras: replace `model.h5` with your Keras model. The app will auto-detect it if TensorFlow is installed.
- For sklearn: replace `model.joblib` with your trained scikit-learn model that implements `predict` and `predict_proba`.
- If you trained a Keras model, also save the scaler as `scaler.joblib` for automatic feature scaling.

Security note:
- This sample is for local development and demonstration only.
