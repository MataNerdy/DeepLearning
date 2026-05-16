# Game of Thrones Survival Prediction — sklearn Classification

## О проекте

Проект посвящён задаче бинарной классификации на табличных данных из вселенной *Game of Thrones*.

Цель — предсказать, выживет ли персонаж (`isAlive`) на основе:

- происхождения,
- дома,
- культуры,
- семейных связей,
- популярности,
- участия в книгах,
- социальных признаков.

![Correlation Matrix](Game_of_thrones/images/correlation_matrix.png)

---

# STAR Summary

## Situation

Исходный датасет содержал:

- большое количество пропусков;
- категориальные признаки высокой кардинальности;
- разреженные genealogical-признаки;
- неоднородные текстовые данные.

Особенно проблемными были:

- `culture`
- `title`
- `house`
- `mother/father/heir/spouse`

Многие признаки имели более 50–80% пропусков.

---

## Task

Необходимо было:

- провести EDA;
- обработать NaN;
- выполнить feature engineering;
- подготовить данные для sklearn-моделей;
- обучить и сравнить несколько алгоритмов классификации;
- выбрать лучшую модель;
- сформировать итоговый `submission.csv`.

---

## Action

В проекте был реализован полный ML pipeline:

### EDA
- анализ распределений;
- исследование пропусков;
- анализ корреляций;
- визуализация survival distribution.

### Feature Engineering
- обработка категориальных признаков;
- создание новых бинарных признаков;
- объединение редких категорий;
- генерация признаков:
  - `isPopular`
  - `boolDeadRelations`
  - `years`

### Encoding
Для категориальных признаков использовался `OneHotEncoder`.

### Обучение моделей
Были протестированы:

- Logistic Regression
- Random Forest
- AdaBoost
- Gaussian Process
- Gaussian Naive Bayes
- KNN
- SVC
- Decision Tree

### Hyperparameter Tuning
Для моделей использовался `GridSearchCV` с `cross-validation`.

---

## Result

Лучший результат показала модель:

# KNeighborsClassifier

| Metric | Value |
|---|---|
| Best CV Accuracy | 0.8369 |
| Holdout Accuracy | **0.8558** |

Лучшие параметры:

```python
{
    'n_neighbors': 21,
    'p': 1,
    'weights': 'distance'
}
