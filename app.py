import streamlit as st
import numpy as np
from model import load_model, predict

# Page config
st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.title("🏥 Breast Cancer Predictor")
st.markdown("Enter or randomize 30 cancer cell features and get a prediction.")

# Load model once (cached)
@st.cache_resource
def get_model():
    return load_model()

# Hardcoded sample data from breast cancer dataset - REAL DATA
SAMPLE_DATA = {
    "Benign Sample 1": np.array([13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]),
    "Benign Sample 2": np.array([13.08, 15.71, 85.63, 520.0, 0.1075, 0.127, 0.04568, 0.0311, 0.1967, 0.06811, 0.1852, 0.7477, 1.383, 14.67, 0.004097, 0.01898, 0.01698, 0.00649, 0.01678, 0.002425, 14.5, 20.49, 96.09, 630.5, 0.1312, 0.2776, 0.189, 0.07283, 0.3184, 0.08183]),
    "Benign Sample 3": np.array([9.504, 12.44, 60.34, 273.9, 0.1024, 0.06492, 0.02956, 0.02076, 0.1815, 0.06905, 0.2773, 0.9768, 1.909, 15.7, 0.009606, 0.01432, 0.01985, 0.01421, 0.02027, 0.002968, 10.23, 15.66, 65.13, 314.9, 0.1324, 0.1148, 0.08867, 0.06227, 0.245, 0.07773]),
    "Malignant Sample 1": np.array([17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]),
    "Malignant Sample 2": np.array([20.57, 17.77, 132.9, 1326.0, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667, 0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.0186, 0.0134, 0.01389, 0.003532, 24.99, 23.41, 158.8, 1956.0, 0.1238, 0.1866, 0.2416, 0.186, 0.275, 0.08902]),
    "Malignant Sample 3": np.array([19.69, 21.25, 130.0, 1203.0, 0.1096, 0.1599, 0.1974, 0.1279, 0.2069, 0.05999, 0.7456, 0.7869, 4.585, 94.03, 0.00615, 0.04006, 0.03832, 0.02058, 0.0225, 0.004571, 23.57, 25.53, 152.5, 1709.0, 0.1444, 0.4245, 0.4504, 0.243, 0.3613, 0.08758]),
}

# Feature names (30 features from breast cancer dataset)
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

# Initialize session state for individual feature keys
if 'features_initialized' not in st.session_state:
    # Initialize with first benign sample
    for i in range(30):
        st.session_state[f"feat_{i}"] = float(SAMPLE_DATA["Benign Sample 1"][i])
    st.session_state.features_initialized = True

model_tuple = get_model()

# Callback function for randomize button
def randomize_features():
    sample_name = np.random.choice(list(SAMPLE_DATA.keys()))
    sample_values = SAMPLE_DATA[sample_name]
    for i in range(30):
        st.session_state[f"feat_{i}"] = float(sample_values[i])

# Layout: Left (inputs), Right (prediction)
col1, col2 = st.columns([2, 1])

with col1:
    # Randomize button with callback
    st.button("🎲 Randomize Data", on_click=randomize_features, key="randomize")

    # Display feature inputs as a grid
    st.subheader("Features")
    feature_cols = st.columns(3)
    for i in range(30):
        col_idx = i % 3
        with feature_cols[col_idx]:
            st.number_input(
                f"{feature_names[i][:12]}",
                value=st.session_state[f"feat_{i}"],
                format="%.4f",
                step=1.0,
                key=f"feat_{i}"
            )

with col2:
    st.subheader("Prediction")
    if st.button("🔮 Predict", key="predict"):
        # Collect all feature values from session state
        X_input = np.array([st.session_state.get(f"feat_{i}", 0.0) for i in range(30)]).reshape(1, -1)
        try:
            proba, pred = predict(model_tuple, X_input)
            st.session_state.proba = proba
            st.session_state.pred = pred
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

    if 'proba' in st.session_state:
        # Note: Model outputs probability of class 1 (Benign in this dataset)
        # pred=1 means Benign, pred=0 means Malignant
        pred_label = "🟢 Benign" if st.session_state.pred == 1 else "🔴 Malignant"
        benign_prob = st.session_state.proba
        malignant_prob = 1.0 - benign_prob
        
        st.metric("Class", pred_label)
        st.metric("Benign Probability", f"{benign_prob:.4f}")
        st.metric("Malignant Probability", f"{malignant_prob:.4f}")
        
        if st.session_state.pred == 1:
            st.success(f"🟢 Benign: {benign_prob*100:.2f}%")
        else:
            st.error(f"🔴 Malignant Risk: {malignant_prob*100:.2f}%")

st.divider()
st.caption("Model: Trained on breast cancer dataset. For demo purposes only.")
