import logging
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from joblib import load
from sklearn.preprocessing import LabelEncoder

from services.datasetService import dataset_completo

logger = logging.getLogger(__name__)


class DiagnosticoService:
    def __init__(self, model_dir: Path | str) -> None:
        self.model_dir = Path(model_dir).resolve()

        model_path = self.model_dir / "modelo_HealthAI.json"
        vectorizer_path = self.model_dir / "vetorizador_HealthAI.pkl"
        encoder_path = self.model_dir / "encoderY_HealthAI.pkl"

        for artifact_path, name in [
            (model_path, "modelo_HealthAI.json"),
            (vectorizer_path, "vetorizador_HealthAI.pkl"),
            (encoder_path, "encoderY_HealthAI.pkl"),
        ]:
            if not artifact_path.exists():
                raise FileNotFoundError(
                    f"Model artifact not found: {artifact_path} ({name})"
                )

        logger.info("Loading XGBoost model from %s", model_path)
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_path))

        logger.info("Loading vectorizer from %s", vectorizer_path)
        vectorizer_loaded = load(str(vectorizer_path))
        if isinstance(vectorizer_loaded, tuple):
            vectorizer_loaded = vectorizer_loaded[0]
        self.vectorizer = vectorizer_loaded

        logger.info("Loading label encoder from %s", encoder_path)
        encoder_data = load(str(encoder_path))
        # Reconstruct the LabelEncoder with the same class ordering used during training
        df = dataset_completo()
        le = LabelEncoder()
        le.fit(df["diagnostico"].astype(str))
        self.label_encoder = le

        logger.info("DiagnosticoService initialized successfully")

    def predict_simple(self, sintomas: list[str] | str) -> dict[str, Any]:
        if isinstance(sintomas, list):
            sintomas_str = " ".join(sintomas)
        else:
            sintomas_str = sintomas

        sintomas_transformed = self.vectorizer.transform([sintomas_str])
        dmatrix = xgb.DMatrix(sintomas_transformed)
        pred_encoded = self.booster.predict(dmatrix)
        if pred_encoded.ndim > 1:
            pred_index = int(pred_encoded[0].argmax())
        else:
            pred_index = int(round(pred_encoded[0]))

        pred_decoded = self.label_encoder.classes_[pred_index]
        return {"diagnostico": pred_decoded}
