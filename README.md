# YandexCup 2023 Nowcasting

## Основная модель
Основная модель - exponential dilated convolutional network (`sources/models/DilConv2`).
Учится и предсказывает только по `intensity`.

Веса после 3 эпох находятся `models/main_model_epoch3.ckpt`.

Метрика по чекпоинту {"public_score": 165.36981108825947, "private_score": 177.62707837007005}

Воспроизведение процесса обучения основной модели

```python main_model_train.py --epochs 3 --batch_size 8 --num_workers 2 ```

## Расширенная модель
Расширенная модель `(sources/models/DilConv2Extend)` = основная + облегченная версия основной, обученная на других фичах.
Учится на `radial_velocity` и `reflectivity` и "исправляет" ошибки основной (а-ля бустинг).

Веса после 2 эпох находятся `models/extend_model_epoch2.ckpt`.

Метрика по чекпоинту {"public_score": 164.70795228819313, "private_score": 177.16333564181045}

Воспроизведение процесса обучения расширенной модели

```python extend_model_train.py --epochs 2 --batch_size 8 --num_workers 2```

Создание файла предсказаний с использованием расширенной модели

```python extend_model_predict.py```

### Расположение данных
Тренировочные в папке train, тестовый файл в корне
