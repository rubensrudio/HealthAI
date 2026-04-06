import logging
from pathlib import Path
from typing import Any

import xgboost as xgb
from joblib import load

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
        self.vectorizer = load(str(vectorizer_path))

        logger.info("Loading label encoder from %s", encoder_path)
        self.label_encoder = load(str(encoder_path))

        logger.info("DiagnosticoService initialized successfully")

    def predict_simple(self, sintomas: list[str] | str) -> dict[str, Any]:
        if isinstance(sintomas, list):
            sintomas_str = " ".join(sintomas)
        else:
            sintomas_str = sintomas

        sintomas_transformed = self.vectorizer.transform([sintomas_str])
        dmatrix = xgb.DMatrix(sintomas_transformed)
        pred_encoded = self.booster.predict(dmatrix)
        pred_index = int(pred_encoded[0])

        pred_decoded = (
            self.label_encoder.classes_[pred_index]
            if hasattr(self.label_encoder, "classes_")
            else str(pred_index)
        )
        return {"diagnostico": pred_decoded}
