# PH2 Skin Lesion Segmentation: SegNet vs U-Net

Проект по бинарной сегментации кожных поражений на дерматоскопических изображениях PH2.
Цель — построить воспроизводимый computer vision pipeline, который выделяет область поражения на медицинском снимке и сравнивает несколько архитектур и функций потерь.

## Situation

В задачах дерматоскопии врачу важно быстро и стабильно выделять границы подозрительного кожного образования. Ручная разметка занимает время и зависит от эксперта, поэтому автоматическая сегментация может быть полезна как вспомогательный этап перед диагностикой, расчетом площади поражения или построением последующей классификационной модели.

Исходный ноутбук был учебным экспериментом по сегментации изображений на датасете PH2. Для GitHub-портфолио он был переработан в проектную структуру со скриптами, CLI-запуском, отдельными модулями моделей, лоссов, метрик и визуализации.

## Task

Нужно было:

- загрузить и подготовить PH2 Dataset;
- привести изображения и маски к размеру 256×256;
- обучить модели SegNet и U-Net для бинарной сегментации;
- сравнить BCE, Dice, Tversky и Focal Loss;
- оценить качество по Dice и IoU;
- сохранить лучший checkpoint;
- визуализировать предсказанные маски;
- оформить проект так, чтобы его можно было запустить не только из ноутбука, но и из командной строки.

## Action

Что реализовано:

- `src/prepare_dataset.py` — подготовка датасета и train/val/test split;
- `src/models.py` — реализации SegNet и U-Net;
- `src/losses.py` — BCE, Dice, Tversky, Focal Loss;
- `src/metrics.py` — Dice Score и IoU;
- `src/train.py` — обучение модели, логирование метрик, сохранение лучшего checkpoint;
- `src/evaluate.py` — оценка на test split;
- `src/visualize_predictions.py` — сохранение примеров: исходное изображение, ground truth mask, prediction;
- `notebooks/Unet_original.ipynb` — исходный исследовательский ноутбук.

## Result

В исходном эксперименте использовался PH2 Dataset: 200 дерматоскопических изображений с масками поражений.

Лучший результат в ноутбуке показала U-Net с Tversky Loss:

| Model | Loss | Val Dice | Val Loss |
|---|---:|---:|---:|
| U-Net | BCE | 0.8657 | 0.4461 |
| U-Net | Dice | 0.8977 | 0.3507 |
| U-Net | Tversky | **0.9004** | **0.2236** |
| U-Net | Focal | 0.8987 | 0.0799 |

Итог: U-Net стабильнее SegNet для этой постановки, а Tversky Loss оказался наиболее полезен для сегментации поражений, где важен баланс между false positive и false negative областями.



Важно: это не медицинский продукт и не диагностическая система. Корректная формулировка для резюме: **research prototype / proof-of-concept для автоматической сегментации кожных поражений**, а не готовый clinical-grade сервис.

## Структура репозитория

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Unet_original.ipynb
├── src/
│   ├── config.py
│   ├── download_data.py
│   ├── prepare_dataset.py
│   ├── dataset.py
│   ├── models.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize_predictions.py
├── data/
│   ├── raw/
│   └── processed/
├── checkpoints/
└── reports/
```

## Быстрый старт

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Скачать архив PH2:

```bash
python src/download_data.py --output-dir data/raw
```

После скачивания архив нужно распаковать так, чтобы структура была примерно такой:

```text
data/raw/PH2Dataset/PH2 Dataset images/...
```

Подготовить `.npy`-массивы:

```bash
python src/prepare_dataset.py \
  --dataset-dir data/raw/PH2Dataset \
  --output-dir data/processed
```

Обучить U-Net с Tversky Loss:

```bash
python src/train.py \
  --model unet \
  --loss tversky \
  --epochs 100 \
  --batch-size 25
```

Оценить модель:

```bash
python src/evaluate.py \
  --model unet \
  --loss tversky
```

Сохранить визуализации предсказаний:

```bash
python src/visualize_predictions.py \
  --model unet \
  --loss tversky \
  --split test \
  --output reports/predictions.png
```

## Стек

- Python
- PyTorch
- TorchVision
- TorchMetrics
- NumPy
- scikit-image
- scikit-learn
- Matplotlib
- tqdm

## Что можно улучшить дальше

- добавить аугментации изображений и масок;
- сделать k-fold cross-validation из-за маленького размера PH2;
- добавить threshold sweep для выбора оптимального порога бинаризации;
- сравнить с pretrained encoder U-Net;
- добавить inference-скрипт для одного изображения;
- завернуть модель в минимальный Streamlit/FastAPI demo;
- логировать эксперименты через MLflow или Weights & Biases.

## Итого

> Разработан воспроизводимый PyTorch-пайплайн для бинарной сегментации кожных поражений на дерматоскопических изображениях PH2: подготовка данных, реализация SegNet/U-Net, сравнение BCE/Dice/Tversky/Focal Loss, оценка по Dice/IoU и визуализация предсказаний. Лучший эксперимент: U-Net + Tversky Loss, validation Dice ≈ 0.90.

Этот проект можно подавать как прикладной ML/CV-прототип для healthcare/medtech-задачи:

- есть понятная бизнес-проблема: ускорить первичную разметку медицинских изображений;
- есть end-to-end pipeline: данные → preprocessing → training → evaluation → visualization;
- есть сравнение архитектур и loss-функций, а не одна случайная модель;
- есть воспроизводимая структура репозитория;
- есть метрики, пригодные для сегментации: Dice и IoU.