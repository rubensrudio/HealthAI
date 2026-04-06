# HealthAI

Machine learning–powered diagnostic assistant that maps symptom descriptions to medical conditions using NLP and gradient boosting.

## Architecture

```
services/           Business logic layer
  datasetService    Curated symptom-diagnosis dataset (21 conditions, 200+ samples)
  vetorizacao       TF-IDF text vectorization + label encoding
  treinamento       XGBoost classifier training & evaluation
  salvarmodelo      Model artifact persistence (PKL/JSON)
  executar          Inference pipeline execution
api/                FastAPI REST interface
  sintomasAPI       Symptom-based diagnosis endpoint
model/              Saved artifacts
  vetorizador       TF-IDF vectorizer
  encoderY          Label encoder
  modelo            Trained XGBoost model
```

## Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| API         | FastAPI + Uvicorn                   |
| ML Model    | XGBoost Classifier                  |
| NLP         | TF-IDF Vectorization (scikit-learn) |
| Data        | pandas                              |
| Persistence | joblib (PKL)                        |

## Diagnoses Covered

21 conditions including: anemia falciforme, artrite reumatoide, diabetes tipo 1, doenca de Alzheimer, doenca de Crohn, doenca de Lyme, doenca de Parkinson, doenca de Wilson, esclerose lateral, esclerose multipla, febre maculosa, fibromialgia, hipertireoidismo, hipotireoidismo, lupus, miastenia gravis, porfiria, sarcoidose, sindrome da fadiga cronica, sindrome de Sjogren.

## Getting Started

### Install dependencies

```bash
apt install python3.12-venv -y && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

### Train and save models

```bash
python model_create.py
```

### Run the API

```bash
uvicorn main:app --reload
```

## Project Structure

```
HealthAI/
  main.py              # FastAPI application entrypoint
  model_create.py      # Training pipeline and model persistence
  requirements.txt     # Python dependencies
  services/            # Modular business logic
  api/                 # REST endpoints
  model/               # Serialized model artifacts
```

## License

Private. All rights reserved.
