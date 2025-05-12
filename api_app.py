from fastapi import FastAPI
from pydantic import BaseModel
import subprocess, uuid, os

app = FastAPI(title="Red‑Wine MLOps API")

class TrainRequest(BaseModel):
    model: str       # elasticnet | ridge | lasso
    alpha: float
    l1_ratio: float | None = None   # ignoré hors ElasticNet

@app.post("/train")
def train(req: TrainRequest):
    """Lance un entraînement MLflow en sous‑processus."""
    run_id = str(uuid.uuid4())[:8]
    cmd = [
        "python", "train_model.py",
        "--model", req.model,
        "--alpha", str(req.alpha)
    ]
    if req.model == "elasticnet":
        cmd += ["--l1_ratio", str(req.l1_ratio or 0.5)]
    # On passe la variable d'env pour que le sous‑processus parle à MLflow
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = env.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    subprocess.Popen(cmd, env=env)
    return {"status": "started", "run_id": run_id, "cmd": " ".join(cmd)}

class PredictRequest(BaseModel):
    alcohol: float
    volatile_acidity: float
    sulphates: float

@app.post("/predict")
def predict(inp: PredictRequest):
    """Exemple minimal de prédiction ‘à la main’. 
       (On chargerait normalement un modèle MLflow.)"""
    # Formule naïve pour démonstration
    score = 3 + 0.3*inp.alcohol - 1.2*inp.volatile_acidity + 0.8*inp.sulphates
    return {"quality_estimate": round(score, 2)}