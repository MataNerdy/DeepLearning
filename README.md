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

# Что демонстрируют проекты

## Machine Learning

- supervised learning,
- binary classification,
- multiclass classification,
- transfer learning,
- boosting,
- hyperparameter tuning,
- cross-validation,
- overfitting analysis.

## Data Science

- EDA,
- feature engineering,
- preprocessing,
- работа с пропусками,
- categorical encoding,
- visualization,
- reproducible pipelines.

## Deep Learning

- CNN,
- EfficientNet,
- ResNet,
- PyTorch training pipelines,
- fine-tuning,
- inference pipelines.

---

# Общий стек технологий

## ML / DL

- Python
- PyTorch
- scikit-learn
- CatBoost

## Data Processing

- pandas
- NumPy

## Visualization

- matplotlib
- seaborn

---

# Структура репозитория

```text
ML_Portfolio/
│
├── Simpsons_Classification/
│
├── Game_of_Thrones_Survival/
│
├── Customer_Churn_Prediction/
│
├── requirements.txt
└── README.md
```

# Основные идеи репозитория

## Репозиторий показывает:

- построение end-to-end ML pipeline;
- работу как с CV, так и с tabular ML;
- preprocessing и feature engineering;
- сравнение baseline и сложных моделей;
- анализ качества моделей и ошибок;
- оформление проектов в reproducible engineering-style формате.

## Возможные дальнейшие улучшения:

- MLflow / experiment tracking,
- Dockerization,
- CI/CD,
- Optuna hyperparameter optimization,
- SHAP explainability,
- ensemble methods,
- deployment inference services,
- API inference pipelines.