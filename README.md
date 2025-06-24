## README.md

### О проекте

Проект представляет собой систему для работы с акустическими сигналами, включающую:

- генерацию синтетических сигналов
- обучение свёрточных нейронных сетей на выбранном датасете
- оценку обучения нейронной сети

---

### Структура проекта

```
.
├── datasets/               # Датасеты в форматах сырых данных и изображений (создаётся в процессе)
├── results/                # Результаты обучения: веса моделей и история обучения (создаётся в процессе)
├── examples/               # Примеры сигналов и результатов обучения 
├── dataset_converter.py    # Конвертация датасета из Matlab формата
├── evaluate.py             # Оценка обученных моделей
├── generator.py            # Генерация синтетических сигналов
├── predict.py              # Модуль предсказаний
├── requirements.txt        # Зависимости проекта
├── train.py                # Обучение моделей
├── utils.py                # Вспомогательные функции
```

---

### Настройка окружения

1. **Требования к Python**:
    * Python версии 3.8 или выше
    * Установленный pip

2. **Установка виртуального окружения**:
   ```bash
   python -m venv venv
   ```

3. **Активация окружения**:
    * Windows:
      ```bash
      venv\Scripts\activate
      ```
    * Linux/MacOS:
      ```bash
      source venv/bin/activate
      ```

4. **Установка зависимостей**:
   ```bash
   pip install -r requirements.txt
   ```

---

### Описание скриптов

#### `dataset_converter.py`

Конвертация датасета из формата Matlab:

```bash
python dataset_converter.py --dataset-path путь_к_датасету --model-number номер_модели
```

Пример:

```shell
python dataset_converter.py --dataset-path "path/to/matlab/dataset" --model-number 6
```

Подробное описания параметров можно получить, выполнив `python dataset_converter.py --help`

---

#### `generator.py`

Генерация синтетических сигналов:

```bash
python generator.py --model-number номер_модели --dataset-size размер_датасета [--interactive-mode]
```

Пример:

```shell
python generator.py --model-number 1 --dataset-size 5000 --interactive-mode
```

Подробное описания параметров можно получить, выполнив `python generate.py --help`

---

#### `train.py`

Обучение модели:

```bash
python train.py --model-number номер_модели --learning-rate скорость_обучения --epochs количество_эпох --batch-size размер_батча --dataset-size размер_датасета
```

Пример:

```shell
python train.py --model-number 1 --learning-rate 0.001 --reduce-learning-rate --epochs 250 --batch-size 32 --dataset-size 5000
```

Подробное описания параметров можно получить, выполнив `python train.py --help`

---

#### `evaluate.py`

Оценка обученной модели:

```bash
python evaluate.py --model-number номер_модели --weights-path путь_к_весам --dataset-size размер_датасета
```

Пример:

```shell
python evaluate.py --model-number 1 --weights-path "results/weights/1/conv1d/6216__2025-05-29T15:52:35__dataset_size=5000__start_lr=0.001__reduce_lr=False__batch_size=32__epochs=250__epoch=0217__val_loss=0.116420.keras" --dataset-size 5000
```

Подробное описания параметров можно получить, выполнив `python evaluate.py --help`  
