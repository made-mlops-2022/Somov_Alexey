import pytest

from src.config.config_classes import TrainingParams
from src.model import get_model


@pytest.mark.parametrize(
    "model_type",
    [
        pytest.param("LogisticRegression"),
        pytest.param("RandomForestClassifier"),
        pytest.param("KNeighborsClassifier"),
    ],
)
def test_get_model(model_type):
    params_raw = dict(
        params={},
        model_type=model_type,
    )
    params = TrainingParams(**params_raw)

    model = get_model(params)

    assert model.__class__.__name__ == model_type