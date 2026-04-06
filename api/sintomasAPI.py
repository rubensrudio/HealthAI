from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path

from services.executarService import DiagnosticoService

router = APIRouter()

_model_dir = Path(__file__).parent.parent / "model"
_diagnostico_service = DiagnosticoService(model_dir=_model_dir)


class SintomasRequest(BaseModel):
    sintomas: str | list[str]


class DiagnosticoResponse(BaseModel):
    diagnostico: str


@router.post("/sintomas", response_model=DiagnosticoResponse)
async def diagnosticar(request: SintomasRequest):
    try:
        result = _diagnostico_service.predict_simple(request.sintomas)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
