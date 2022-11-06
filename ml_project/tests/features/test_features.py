import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer

from src.config.config_classes import FeatureTransformer
from src.features import build_transformer, make_features


@pytest.mark.parametrize(
    "params, result",
    [
        ([dict(trans_name="num", trans_class='StandardScaler', column_names=[0, 1],
               params={'with_mean': False})],
         True
         ),
        ([dict(trans_name="num", trans_class='SimpleImputer', column_names=[0, 1], params={}),
          dict(trans_name="num", trans_class='StandardScaler', column_names=[0, 1], params={})],
         True
         ),
        ([dict(trans_name="num", trans_class='CustomTransformer', column_names=[0, 1], params={})],
         False
         )
    ],
)
def test_build_transform(params, result):
    transform_params = [FeatureTransformer(**t_param) for t_param in params]
    if result:
        transformer = build_transformer(transform_params)
        assert len(transformer._transformers) == len(params)
    else:
        with pytest.raises(AttributeError):
            build_transformer(transform_params)


def test_make_features():
    transformer = ColumnTransformer(
        [("norm1", Normalizer(norm='l1'), [0, 1]),
         ("norm2", Normalizer(norm='l1'), slice(2, 4))])

    data_df = pd.DataFrame([[0., 1., 2., 2.], [1., 1., 0., 1.]])
    transformer.fit(data_df)

    result_df = pd.DataFrame([[0., 1., 0.5, 0.5], [0.5, 0.5, 0., 1.]])
    df_features = make_features(transformer, data_df)

    assert df_features.values.tolist() == result_df.values.tolist()