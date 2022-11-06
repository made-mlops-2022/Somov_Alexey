import yaml

from src.config import read_config


def test_build_config(tmp_path, config_dict):
    path = tmp_path / "test_config.yml"
    with open(path, "w") as fd:
        yaml.dump(config_dict, fd)

    config = read_config(path)

    assert config.logger_config_path == config_dict['logger_config_path']
    assert config.input_data_path == config_dict['input_data_path']
    assert config.test_data_path == config_dict['test_data_path']
    assert config.predict_path == config_dict['predict_path']
    assert config.output_model_path == config_dict['output_model_path']
    assert config.metric_path == config_dict['metric_path']
    assert config.split_params.val_size == config_dict['split_params']['val_size']
    assert config.split_params.random_state == config_dict['split_params']['random_state']
    assert config.train_params.params == config_dict['train_params']['params']
    assert config.train_params.model_type == config_dict['train_params']['model_type']
    assert config.feature_params.cat_features == config_dict['feature_params'].get('cat_features', None)
    assert config.feature_params.num_features == config_dict['feature_params'].get('num_features', None)
    assert config.feature_params.features_to_drop == config_dict['feature_params'].get('features_to_drop', None)
    assert config.feature_params.target_col == config_dict['feature_params'].get('target_col', None)
    assert config.feature_params.target_col == config_dict['feature_params']['target_col']

    transformers = config_dict['feature_params'].get('transform_params', None)
    if transformers is not None:
        assert len(config.feature_params.transform_params) == len(transformers)