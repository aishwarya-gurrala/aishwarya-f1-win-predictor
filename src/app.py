import streamlit as st
import shap
import matplotlib.pyplot as plt
from model_utils import load_model, prepare_input

st.set_page_config(page_title="Aishwarya's F1 Predictor", layout="centered")
st.title("🏁 Formula 1 Win Probability Predictor")

grid = st.sidebar.number_input("Starting Grid Position", 1, 30, 5)
points = st.sidebar.slider("Driver Points", 0, 500, 100)

input_df = prepare_input(grid, points)
model = load_model()

if st.button("Predict"):
    try:
        prob = model.predict_proba(input_df)[0][1]
        st.subheader(f"🏁 Win Probability: {round(prob * 100, 2)}%")
    except Exception as ex:
        st.error(f"Prediction failed: {ex}")
        st.stop()

    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    st.markdown("### 🔍 Feature Impact (SHAP)")

    # Create an explicit figure so Streamlit can render it safely
    fig, ax = plt.subplots()
    # Note: shap.plots.waterfall does not accept an `ax=` argument in this version,
    # so we rely on it drawing into the current figure.
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig, bbox_inches="tight")
