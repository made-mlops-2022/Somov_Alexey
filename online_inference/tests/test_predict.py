import json
import pytest

from fastapi.testclient import TestClient
from online_inference.app import app


@pytest.mark.parametrize(
    "data",
    (
            {},
            {'wrong': 1},
            {"age": 60, "sex": 1, "cp": 0, "trestbps": 130, "chol": 253, "fbs": 0, "restecg": 1,
             "thalach": 144, "exang": 1, "oldpeak": 1.4, "slope": 2, "ca": 1, "thal": 3
             },
            {"idx": 1, "age": 1000, "sex": 1, "cp": 0, "trestbps": 130, "chol": 253, "fbs": 0, "restecg": 1,
             "thalach": 144, "exang": 1, "oldpeak": 1.4, "slope": 2, "ca": 1, "thal": 3
             },
            {"idx": "wrong", "age": 60, "sex": 1, "cp": 0, "trestbps": 130, "chol": 253, "fbs": 0, "restecg": 1,
             "thalach": 144, "exang": 1, "oldpeak": 1.4, "slope": 2, "ca": 1, "thal": 3
             },
    ))
def test_predict_wrong_data(data):
    with TestClient(app) as test_client:
        response = test_client.post("/predict", data=json.dumps(data))
        assert 400 <= response.status_code < 500


def test_predict_correct_data(fake_dataset):
    with TestClient(app) as test_client:
        for data in fake_dataset:
            response = test_client.post("/predict", data=data)
            assert response.status_code == 200