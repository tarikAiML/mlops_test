import streamlit as st
import requests, json, pandas as pd

st.set_page_config(page_title="Red‑Wine Trainer", layout="wide")

st.title("🍷 Red‑Wine Trainer – Interface Streamlit")
st.markdown(
    "Lancez des entraînements MLflow et observez‑les en temps réel dans "
    "[MLflow UI](http://localhost:5000).")

with st.form("train_form"):
    model = st.selectbox("Modèle", ["elasticnet", "ridge", "lasso"])
    alpha = st.slider("alpha", 0.01, 2.0, 0.5, 0.01)
    l1_ratio = st.slider("l1_ratio (ElasticNet)", 0.0, 1.0, 0.5, 0.05)
    submitted = st.form_submit_button("🚀 Lancer l'entraînement")
    if submitted:
        payload = {"model": model, "alpha": alpha, "l1_ratio": l1_ratio}
        resp = requests.post("http://api:8000/train",
                             data=json.dumps(payload),
                             headers={"Content-Type": "application/json"})
        if resp.ok:
            st.success(f"Run démarré : {resp.json()['run_id']}")
        else:
            st.error("Erreur lors de l’appel API")

st.header("Prévision rapide (démonstration)")
col1, col2, col3 = st.columns(3)
with col1: alcohol   = st.number_input("Alcohol",  8.0, 15.0, 10.0, 0.1)
with col2: vola_acid = st.number_input("Volatile acidity", 0.1, 1.5, 0.5, 0.01)
with col3: sulphates = st.number_input("Sulphates", 0.3, 1.8, 0.8, 0.05)

if st.button("🔮 Prédire qualità"):
    payload = {"alcohol": alcohol,
               "volatile_acidity": vola_acid,
               "sulphates": sulphates}
    r = requests.post("http://api:8000/predict",
                      data=json.dumps(payload),
                      headers={"Content-Type": "application/json"})
    st.write(r.json())