import os
import joblib
import xgboost as xgb
from services.datasetService import dataset_completo

class DiagnosticoService:
    def __init__(self, pathModel: str):
        self.pathRootModel = pathModel
        self.HealthAI = xgb.XGBClassifier()
        self.HealthAI.load_model(os.path.join(self.pathRootModel, "modelo_HealthAI.json"))
        obj = joblib.load(os.path.join(self.pathRootModel, "vetorizador_HealthAI.pkl"))
        self.vetorizador = obj[0] if isinstance(obj, tuple) else obj
        self.encode_Y = joblib.load(os.path.join(self.pathRootModel, "encoderY_HealthAI.pkl"))
        
    def predict_simple(self, sintomas):
        if isinstance(sintomas, list):
            sintomas_str = " ".join(sintomas)
        else:
            sintomas_str = sintomas
        sintomas_transform = self.vetorizador.transform([sintomas_str])
        predicao_encoded = self.HealthAI.predict(sintomas_transform)
        predicao_decoded = self.encode_Y.classes_[int(predicao_encoded[0])] if hasattr(self.encode_Y, 'classes_') else str(predicao_encoded[0])
        LabelY = dataset_completo()["diagnostico"].astype(str).unique()
        label_pred = LabelY[int(predicao_decoded[0])]
        return label_pred
        