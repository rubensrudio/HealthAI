from fastapi import APIRouter
from services.executarService import DiagnosticoService

router = APIRouter()
diagnosticoAI = DiagnosticoService(pathModel="model")

@router.get("/")
async def get_root():
    return {"message": "Welcome to the sintomas API!"}  

@router.get("/predict/")
async def predict(sintomas: str):
    sintomas_list = sintomas.split(",")
    resultado = diagnosticoAI.predict_simple(sintomas_list)
    return {
        "sintomas": sintomas_list,
        "diagnostico": resultado
    }