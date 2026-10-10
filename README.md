# TextClassifer3000

A Wikipedia article topic classifier using TF-IDF and Multinomial Naive Bayes,
with a FastAPI prediction API. Categories: Technology, History, and Science.
    
## Setup and Training

Use Python 3.10 or newer. Run all commands from the repository root so the
training scripts save their artifacts where the API expects them.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/data_collector.py
python src/preprocessor.py
python src/model_trainer.py
```

The collector needs internet access. If you already have the dataset and trained
artifacts, you can skip those steps. Use the same scikit-learn environment for
training and serving; retrain when upgrading scikit-learn. Only load trusted
joblib files.

## Start the API

```bash
python -m uvicorn src.api:app --reload
```

Open http://127.0.0.1:8000/docs to try the API in your browser. Expand
`POST /predict`, select **Try it out**, enter the JSON below, and click **Execute**.

```json
{"text": "Computer software and artificial intelligence are changing technology."}
```

Or use curl:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Computer software and artificial intelligence are changing technology."}'
```

Responses contain `category` and a `probabilities` object mapping each category
to its model-estimated probability. These values are not guaranteed accuracy
or calibrated confidence. The dataset is very small, so results are experimental.

`GET /health` returns `{"status": "ok"}` after successful startup. Invalid,
blank, over-50,000-character, and out-of-vocabulary text returns HTTP 422.

## How it works

`src/api.py` loads `text_classification_model.pkl` and `tfidf_vectorizer.pkl`
from the repository root once at application startup. Missing artifacts stop
startup with instructions to train first. Requests use the shared `clean_text`
function, the saved vectorizer's `transform`, and the saved model's
`predict_proba`. Neither training nor data collection runs inside the API.

`src/text_cleaning.py` is shared with preprocessing to keep training and
prediction text handling consistent. Existing training commands still work.

`--reload` is for local development. Restart the API after retraining to load
the new artifacts.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests train a tiny local model to verify API behavior, preprocessing consistency,
input validation, and missing-artifact errors without fetching Wikipedia data.
