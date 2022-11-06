# MLOps hw1

## Настройка окружения
* `pip3 install virtualenv`
* `virtualenv mlops_venv` 
* `source mlops_venv/bin/activate`
* `pip install requirements.txt`

## Обучение моделей

```bash
python ml_project/train.py --config-name ./configs/model_1.yml
python ml_project/train.py --config-name ./configs/model_2.yml
```

## Предсказание

```bash
python ml_project/predict.py --config-name ./configs/model_1.yml
python ml_project/predict.py --config-name ./configs/model_2.yml
```

## Тестирование
```bash
 python -m pytest -v .
```

## Проверка с помощью pylint
```bash
pylint ml_project --disable=C0114,C0115,C0116 --fail-under=7.0
```
