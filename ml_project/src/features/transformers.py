import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


SimpleImputer = SimpleImputer
StandardScaler = StandardScaler
OneHotEncoder = OneHotEncoder


class Log1p(BaseEstimator, TransformerMixin):
    def __init__(self, use_1p=False):
        self.use_1p = use_1p

    def transform(self, X, y=None):
        Xo = self._validate_data(X, reset=False)

        Xo -= np.min(Xo, axis=0, keepdims=True)
        if self.use_1p:
            X_transformed = np.log(Xo)
        else:
            X_transformed = np.log1p(Xo)

        return X_transformed
