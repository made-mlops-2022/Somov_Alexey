# hw2

## Датасет
Установить папку artefacts по ссылке [google-disk](https://drive.google.com/file/d/1seYrXfnFOXTMARkGz3qEBhI7oCKsVGqL/view?usp=sharing)
```bash
wget --no-check-certificate 'https://drive.google.com/file/d/1seYrXfnFOXTMARkGz3qEBhI7oCKsVGqL/view?usp=sharing' -O artefacts.zip && unzip artefacts.zip
```

## Установка виртуального окружения
* `pip3 install virtualenv`
* `virtualenv venv_online` 
* `source venv_online/bin/activate`
* `pip install .`

## Запуск

```bash
MODEL_PATH='artefacts/model.pkl' uvicorn online_inference.app:app
```

## Запросы

```bash
python online_inference/request.py --config-name='configs/request.yml'
```

## Докер
##### Оптимизация размера образа (сжат до 206.64 MB)
* Исходный размер - 1.29 Gb
* python:3.9-slim - 522 Mb
* light_requirements.txt - 494 Mb

### Локальный запуск контейнера
```bash
docker build -t online .
docker run -p 8000:8000 --rm online
```

### DockerHub
```bash
docker pull alsomov/online_inference
docker run -p 8000:8000 --rm alsomov/online_inference
```

## Тестирование
```bash
MODEL_PATH=artefacts/model.pkl pytest -v .
```

## Линтер
```bash
pylint online_inference --disable=C0114,C0115,C0116 --fail-under=7.0
```
