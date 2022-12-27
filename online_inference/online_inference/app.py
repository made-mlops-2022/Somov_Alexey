import os
import logging

import pickle
from fastapi import FastAPI, status, Response
from fastapi.responses import RedirectResponse
import uvicorn

from online_inference.config.config_classes import QueryData, Prediction

logger = logging.getLogger("uvicorn")
app = FastAPI()
app.MODEL = None


@app.get('/')
def root():
    return RedirectResponse(app.docs_url)


@app.get('/health')
def health():
    return bool(app.MODEL)


@app.on_event("startup")
def prepare_model():
    model_path = os.getenv("MODEL_PATH")
    logger.info("load model from %s", model_path)

    try:
        with open(model_path, "rb") as fd:
            app.MODEL = pickle.load(fd)
    except Exception:
        raise RuntimeError(f"Can't load model from {model_path}")


@app.post('/predict')
async def predict(data: QueryData, response: Response):
    if not health():
        logger.warning("fail prediction model not ready")
        response.status_code = status.HTTP_425_TOO_EARLY
        return None

    logger.info("make prediction for query: %s", data)
    features = list(data.dict().values())[1:]
    y_pred = app.MODEL.predict_proba([features])[0, 1]
    prediction = Prediction(id=data.idx, y_pred=y_pred)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app=app)