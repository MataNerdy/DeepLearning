# Game of Thrones Survival Prediction — sklearn Classification

### Situation

Решение задачи бинарной классификации на датасете персонажей вселенной «Игры престолов».

Датасет содержал большое количество пропусков, категориальных признаков и неоднородных текстовых данных:

* дома (`House`)
* титулы (`Title`)
* культуры (`Culture`)
* связи между персонажами
* демографические признаки

Цель — предсказать, выживет ли персонаж (`isAlive`).

---

### Task

Нужно было:

* провести полный EDA и предобработку данных;
* обработать большое количество NaN;
* спроектировать новые признаки;
* подготовить категориальные данные для ML-моделей;
* обучить и сравнить несколько алгоритмов классификации из `sklearn`;
* выбрать лучшую модель по accuracy;
* сформировать финальный `submission.csv`.

---

### Action

В рамках проекта был реализован полный tabular ML pipeline:

#### 1. Exploratory Data Analysis (EDA)

* анализ распределений признаков;
* визуализация пропусков;
* анализ баланса классов;
* исследование корреляций;
* анализ влияния домов, титулов, культур и популярности на выживаемость.

#### 2. Data Cleaning & Feature Engineering

* обработка NaN;
* объединение редких категорий;
* нормализация названий культур;
* создание бинарных признаков:
  * `isPopular`
  * `boolDeadRelations`
* извлечение дополнительной информации из имен персонажей;
* инженерия признака `years` на основе возраста и даты рождения.

#### 3. Encoding

Для категориальных признаков использовался `OneHotEncoder`, поскольку между категориями отсутствует отношение порядка.

#### 4. Model Training

Были обучены и сравнены:

* Logistic Regression
* Random Forest
* AdaBoost
* Gaussian Process
* Gaussian Naive Bayes
* KNN
* SVC
* Decision Tree

#### 5. Evaluation

Модели сравнивались по метрике `accuracy`.

Лучший результат на holdout-выборке показала модель `KNeighborsClassifier` с accuracy = 0.8558.

---

### Result

Проект успешно прошел валидацию на Stepik.

Ключевые результаты:

* реализован полный ML pipeline для табличной классификации;
* проведен feature engineering для сложных категориальных данных;
* протестировано несколько ML-алгоритмов;
* получен production-like опыт сравнения моделей;
* оформлен финальный submission для leaderboard.

---

# 🧠 Технологии

* Python
* pandas
* NumPy
* matplotlib
* seaborn
* scikit-learn

---

# 📂 Структура репозитория

```text
project/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── submission.csv
│
├── notebooks/
│   └── Stepik_hw1.ipynb
│
├── images/
│   ├── nan_analysis.png
│   ├── survival_by_house.png
│   ├── correlation_matrix.png
│   └── leaderboard_score.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🔍 Что интересно в проекте

## Dataset Overview

Исходный датасет содержал 1557 объектов и 25 признаков.

### Типы данных

* числовые признаки (`age`, `popularity`, `numDeadRelations`)
* бинарные признаки (`male`, `isNoble`, `isMarried`)
* категориальные признаки (`house`, `culture`, `title`)
* сильно разреженные genealogical-признаки (`mother`, `father`, `heir`, `spouse`)

### Проблема пропусков

Одной из главных сложностей проекта стало большое количество missing values.

#### Train dataset

* `title` — 53.95% NaN
* `house` — 24.47% NaN
* `culture` — 68.66% NaN
* `age` — 82.08% NaN

#### Test dataset

* `title` — 43.19% NaN
* `house` — 11.83% NaN
* `culture` — 51.41% NaN
* `age` — 60.41% NaN

Это потребовало аккуратной обработки категориальных признаков и feature engineering.

## Feature Engineering

Главная часть проекта — не обучение моделей, а именно работа с признаками.

Особенно интересными оказались:

* агрессивная очистка категориальных признаков;
* объединение редких культур и домов;
* попытка восстановить отсутствующие данные через лор и имена персонажей;
* создание бинарных признаков поверх noisy data.

## Работа с категориальными данными

Проект хорошо демонстрирует:

* проблемы high-cardinality категорий;
* влияние One-Hot Encoding;
* последствия редких категорий для ML-моделей.

## Результаты моделей

| Model               | Best CV Accuracy | Holdout Accuracy |
| ------------------- | ---------------- | ---------------- |
| Logistic Regression | 0.8506           | 0.8397           |
| Random Forest       | 0.8506           | 0.8494           |
| AdaBoost            | 0.8554           | 0.8429           |
| Gaussian Process    | 0.8490           | 0.8429           |
| GaussianNB          | 0.8209           | 0.8333           |
| KNN                 | 0.8369           | **0.8558**       |
| SVC                 | 0.8522           | 0.8462           |
| Decision Tree       | 0.8530           | 0.8397           |

### Лучшие гиперпараметры

#### KNN

```python
{
    'n_neighbors': 21,
    'p': 1,
    'weights': 'distance'
}
```

#### Random Forest

```python
{
    'max_depth': 10,
    'max_features': 'sqrt',
    'min_samples_split': 2,
    'n_estimators': 100
}
```

#### AdaBoost

```python
{
    'estimator__max_depth': 2,
    'learning_rate': 1.0,
    'n_estimators': 200
}
```

## Сравнение классических ML-моделей

Проект показывает различия между:

* линейными моделями;
* деревьями решений;
* ансамблями;
* kernel-based моделями.

---

# 📈 Возможные улучшения

Что можно было бы сделать дальше:

* добавить cross-validation;
* использовать GridSearchCV;
* попробовать CatBoost / XGBoost;
* добавить feature importance analysis;
* провести calibration analysis;
* попробовать target encoding;
* оформить pipeline через sklearn Pipeline.

---

# 🚀 Как запустить

## Установка

```bash
pip install -r requirements.txt
```

## Запуск ноутбука

```bash
jupyter notebook
```

Открыть:

```text
notebooks/Stepik_hw1.ipynb
```

---

# 💡 Чему я научилась в проекте

* работать с табличными ML-задачами;
* проводить EDA;
* проектировать признаки;
* обрабатывать категориальные данные;
* сравнивать ML-модели;
* выстраивать базовый ML workflow;
* оформлять результаты для leaderboard-задач.

---

# 📬 Контакты

GitHub: [https://github.com/MataNerdy](https://github.com/MataNerdy)
