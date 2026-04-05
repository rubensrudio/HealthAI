import logging
from pathlib import Path
from typing import Any

import joblib
import xgboost as xgb

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
        self.health_ai = xgb.XGBClassifier()
        self.health_ai.load_model(str(model_path))

        logger.info("Loading vectorizer from %s", vectorizer_path)
        vectorizer_obj = joblib.load(str(vectorizer_path))
        self.vectorizer = vectorizer_obj[0] if isinstance(vectorizer_obj, tuple) else vectorizer_obj

        logger.info("Loading label encoder from %s", encoder_path)
        self.label_encoder = joblib.load(str(encoder_path))

        logger.info("DiagnosticoService initialized successfully")

    def predict_simple(self, sintomas: list[str] | str) -> dict[str, Any]:
        if isinstance(sintomas, list):
            sintomas_str = " ".join(sintomas)
        else:
            sintomas_str = sintomas

        sintomas_transformed = self.vectorizer.transform([sintomas_str])
        predicao_encoded = self.health_ai.predict(sintomas_transformed)

        pred_decoded = (
            self.label_encoder.classes_[int(predicao_encoded[0])]
            if hasattr(self.label_encoder, "classes_")
            else str(predicao_encoded[0])
        )
        return {"diagnostico": pred_decoded}
