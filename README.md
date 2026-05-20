# ML & Deep Learning Portfolio

Репозиторий c pet-проектами по Machine Learning, Deep Learning и Data Science.

Проекты охватывают:

- computer vision,
- tabular machine learning,
- feature engineering,
- transfer learning,
- hyperparameter tuning,
- exploratory data analysis,
- reproducible ML pipelines.

Основной фокус — не только получение хороших метрик, но и построение понятных end-to-end ML решений с полноценным анализом данных, сравнением моделей и оформлением результатов.

---

# Проекты

## 1. Simpsons Character Classification (Computer Vision)

Классификация персонажей мультсериала Simpsons с помощью CNN и transfer learning на PyTorch.

### Что реализовано

- custom CNN architecture,
- transfer learning,
- fine-tuning pretrained моделей,
- image preprocessing pipeline,
- DataLoader + augmentations,
- inference pipeline,
- визуализация предсказаний,
- сравнение CNN-архитектур.

### Использованные модели

- SimpleCNN
- ResNet18
- EfficientNet-B0

### Лучший результат

| Metric | Value |
|---|---|
| Validation Accuracy | **0.9631** |
| Kaggle Score | **0.98618** |

### Основные ML-инсайты

- transfer learning существенно ускоряет convergence;
- EfficientNet-B0 дал лучший accuracy/efficiency tradeoff;
- fine-tuning pretrained моделей улучшил качество более чем на 13% относительно baseline CNN.

### Tech Stack

- Python
- PyTorch
- torchvision
- NumPy
- pandas
- matplotlib

---

## 2. Game of Thrones Survival Prediction (Tabular ML)

Бинарная классификация на табличных данных: предсказание выживания персонажей Game of Thrones с помощью sklearn.

### Что реализовано

- EDA,
- feature engineering,
- обработка high-cardinality категорий,
- работа с пропусками,
- One-Hot Encoding,
- GridSearchCV,
- сравнение sklearn-моделей,
- modular ML pipeline.

### Протестированные модели

- Logistic Regression
- Random Forest
- AdaBoost
- Gaussian Process
- GaussianNB
- KNN
- SVC
- Decision Tree

### Лучший результат

| Metric | Value |
|---|---|
| Best Holdout Accuracy | **0.8558** |

### Основные ML-инсайты

- feature engineering критически влияет на качество tabular ML;
- distance-based методы могут outperform сложные модели после хорошей обработки признаков;
- корректная обработка категориальных признаков и NaN существенно повышает accuracy.

### Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn

---

## 3. Customer Churn Prediction (Business ML)

ML-проект по предсказанию оттока клиентов с использованием tabular ML и ROC-AUC optimization.

### Что реализовано

- полноценный EDA,
- preprocessing pipeline,
- обработка категориальных признаков,
- baseline linear models,
- CatBoost training,
- cross-validation,
- hyperparameter tuning,
- ROC-AUC evaluation,
- анализ переобучения.

### Использованные модели

- Logistic Regression
- CatBoostClassifier

### Лучший результат

| Metric | Value |
|---|---|
| Validation ROC-AUC | **0.8548** |

### Основные ML-инсайты

- более сложная модель не всегда дает лучшее качество;
- линейные модели могут outperform boosting на небольших tabular datasets;
- корректный preprocessing и feature handling часто важнее complexity модели.

### Business Insights

EDA показал:

- churn значительно выше у month-to-month клиентов;
- long-term contracts улучшают retention;
- internet service type сильно влияет на churn probability.

### Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- CatBoost
- matplotlib
- seaborn

---
## 4. Halo Infinite Object Detection

Современный object detection pipeline на PyTorch с исследованием detector architectures, assignment strategies и IoU-based losses.

### Что реализовано

- custom PyTorch detector,
- EfficientNet backbone,
- FPN / PAN neck,
- decoupled detection head,
- TAL label assignment,
- DIoU loss,
- YOLOv8 baseline comparison.

### Лучшие результаты

#### Custom Detector

| Metric | Value |
|---|---|
| Validation mAP | 0.1545 |

#### YOLOv8 Baseline

| Metric | Value |
|---|---|
| Precision | 0.945 |
| Recall | 0.854 |
| mAP50 | 0.915 |
| mAP50-95 | 0.6698 |

### Основные ML-инсайты

- TAL + DIoU дали наиболее заметный прирост качества;
- assignment strategy и loss balancing оказались важнее backbone architecture;
- production-grade training recipes существенно влияют на качество detector'ов.

### Ключевые технологии

- PyTorch
- YOLOv8
- Albumentations
- FPN / PAN
- TAL
- DIoU Loss

---

## 5. Skin Lesion Segmentation with U-Net and SegNet

Proof-of-concept CV-система для автоматической сегментации кожных поражений на дерматоскопических изображениях.

### Что реализовано

- SegNet и U-Net,
- BCE / Dice / Tversky / Focal Loss,
- Dice и IoU evaluation,
- preprocessing pipeline,
- visualization pipeline,
- experiment comparison.

### Лучший результат

| Model | Loss | Validation Dice |
|---|---|---|
| U-Net | Tversky Loss | **0.900** |

### Основные ML-инсайты

- U-Net существенно превосходит SegNet благодаря skip-connections;
- Tversky Loss оказался наиболее устойчивым для lesion segmentation;
- BCE Loss хуже оптимизирует форму медицинских масок.

### Ключевые технологии

- PyTorch
- U-Net
- SegNet
- Medical Imaging
- Dice / IoU
- Semantic Segmentation

---

## 6. Text-Guided Face Editing with StyleGAN2, CLIP and ArcFace

Generative AI / Computer Vision проект по text-guided редактированию синтетических лиц.

### Что реализовано

- StyleGAN2 generation,
- CLIP-guided optimization,
- ArcFace identity preservation,
- latent optimization,
- ID Loss,
- grid search коэффициентов.

### Основная задача

Редактирование изображения по текстовому промпту:

```text
a person with bright green hair
```

при сохранении identity исходного лица.

### Основные ML-инсайты

- высокий ID Loss лучше сохраняет лицо, но ослабляет редактирование;
- низкая регуляризация усиливает эффект, но увеличивает артефакты;
- баланс между CLIP guidance и identity preservation критичен для controllable editing.

### Ключевые технологии

- StyleGAN2
- OpenAI CLIP
- ArcFace
- Latent Optimization
- Generative AI
- PyTorch

---

# Что демонстрируют проекты

## Computer Vision

- image classification,
- object detection,
- semantic segmentation,
- generative image editing,
- multi-scale feature extraction,
- medical image analysis.

## Machine Learning

- supervised learning,
- binary classification,
- multiclass classification,
- boosting,
- hyperparameter tuning,
- cross-validation,
- overfitting analysis.

## Deep Learning

- CNN,
- EfficientNet,
- ResNet,
- U-Net,
- StyleGAN2,
- ArcFace,
- transfer learning,
- fine-tuning.

## Data Science

- EDA,
- feature engineering,
- preprocessing,
- categorical encoding,
- reproducible ML pipelines,
- visualization,
- business metrics analysis.

---

# Общий стек технологий

## ML / DL

- Python
- PyTorch
- scikit-learn
- CatBoost
- Ultralytics YOLO

## Data Processing

- pandas
- NumPy

## Visualization

- matplotlib
- seaborn

## CV / DL Tools

- Albumentations
- torchvision
- TorchMetrics
- OpenAI CLIP

---

# Структура репозитория

```text
ML_Portfolio/
│
├── Simpsons_Classification/
├── Game_of_Thrones_Survival/
├── Customer_Churn_Prediction/
├── Halo_Object_Detection/
├── Skin_Lesion_Segmentation/
├── Text_Guided_Face_Editing/
│
├── requirements.txt
└── README.md
```

---

# Основные идеи репозитория

Репозиторий показывает:

- построение end-to-end ML pipeline;
- работу с CV, tabular ML и generative AI;
- preprocessing и feature engineering;
- сравнение baseline и production-grade решений;
- анализ качества моделей и ошибок;
- reproducible engineering-style подход;
- исследовательский подход к ML experimentation.

---

# Возможные дальнейшие улучшения

- MLflow / experiment tracking,
- Dockerization,
- CI/CD,
- Optuna hyperparameter optimization,
- SHAP explainability,
- ensemble methods,
- deployment inference services,
- FastAPI / Streamlit demos,
- distributed training.

---

# GitHub

GitHub Profile: https://github.com/MataNerdy
