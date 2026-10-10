from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from .clean_text import clean_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class PredictionRequest(BaseModel):
    text: str = Field(..., description="The text to classify.")

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value) -> str:
        if not value.strip():
            raise ValueError("Text cannot be empty.")
        return value

class PredictionResponse(BaseModel):
    category: str
    probabilities: dict[str, float]

def create_app(artifact_dir: Path = PROJECT_ROOT) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load model and vectorizer from disk during startup
        directory = Path(artifact_dir)
        model_path = directory / "text_classification_model.pkl"
        vectorizer_path = directory / "tfidf_vectorizer.pkl"
        for path in (model_path, vectorizer_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}. run the collector, preprocessor, and trainer from the repository root before starting the API.")
        app.state.model = joblib.load(model_path)
        app.state.vectorizer = joblib.load(vectorizer_path)
        yield  # This is required for the asynccontextmanager to work
        del app.state.model
        del app.state.vectorizer

    app = FastAPI(title="TestClassifier3000 API", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(body: PredictionRequest, request: Request) -> PredictionResponse:
        vectorizer = request.app.state.vectorizer
        model = request.app.state.model
        features = vectorizer.transform([clean_text(body.text)])
        if features.nnz == 0:
            raise HTTPException(status_code=422,
                                 detail="Text contains no words in the model's learned vocabulary.")
        probabilities = model.predict_proba(features)[0]
        return PredictionResponse(
            category=str(model.classes_[probabilities.argmax()]),
            probabilities={
                str(label): float(probability) 
                for label, probability in zip(model.classes_, probabilities)
            },
        )
    return app

app = create_app()