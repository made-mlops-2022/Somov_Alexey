import pytest

@pytest.fixture()
def faker_seed():
    return 42

@pytest.fixture
def fake_dataset(faker):
    data = [faker.json(
        data_columns=[
            ('idx', 'pyint', {'min_value': 50, 'max_value': 100}),
            ('age', 'pyint', {'min_value': 0, 'max_value': 120}),
            ('sex', 'pyint', {'min_value': 0, 'max_value': 1}),
            ('cp', 'pyint', {'min_value': 0, 'max_value': 3}),
            ('trestbps', 'pyint', {'min_value': 20, 'max_value': 250}),
            ('chol', 'pyint', {'min_value': 0, 'max_value': 1000}),
            ('fbs', 'pyint', {'min_value': 0, 'max_value': 1}),
            ('restecg', 'pyint', {'min_value': 0, 'max_value': 2}),
            ('thalach', 'pyint', {'min_value': 20, 'max_value': 250}),
            ('exang', 'pyint', {'min_value': 0, 'max_value': 1}),
            ('oldpeak', 'pyfloat', {'min_value': 0.0, 'max_value': 10.0}),
            ('slope', 'pyint', {'min_value': 0, 'max_value': 2}),
            ('ca', 'pyint', {'min_value': 0, 'max_value': 4}),
            ('thal', 'pyint', {'min_value': 0, 'max_value': 3}),
        ], num_rows=1
    ).replace("[", "").replace("]", "") for _ in range(100)]

    return data