from api.sintomasAPI import router as sintomas_router
from fastapi import FastAPI

app = FastAPI(title="HealthAI API", version="1.0.0", description="API for HealthAI application")
app.include_router(sintomas_router, tags=["sintomas"])