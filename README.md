# ML & Deep Learning Portfolio

Репозиторий c pet-проектами по Machine Learning, Deep Learning и Data Science.

Проекты охватывают:

- Computer Vision,
- NLP,
- Retrieval & RAG,
- Semantic Search,
- Generative AI,
- Tabular ML,
- Deep Learning,
- Feature Engineering,
- Transfer Learning,
- Hyperparameter Tuning,
- Reproducible ML Pipelines.

Основной фокус — не только получение хороших метрик, но и построение понятных end-to-end ML решений с полноценным анализом данных, сравнением моделей и инженерным оформлением экспериментов.

---

# Portfolio Overview

| Project | Domain | Main Tech | Best Metric |
|---|---|---|---|
| Simpsons Classification | CV | EfficientNet-B0 | Acc 0.9631 |
| Halo Object Detection | Object Detection | YOLOv8 | mAP50 0.915 |
| Skin Lesion Segmentation | Segmentation | U-Net | Dice 0.900 |
| Tourism RAG Assistant | RAG / NLP | ColBERT + Mistral | Relevancy 0.85 |
| Semantic Search | Retrieval | Word2Vec | Hits@10 0.641 |
| Math Topic Classification | NLP | RuBERT | Acc 0.6206 |
| NLP Text Classification | NLP | GRU | Acc 0.9176 |
| Customer Churn Prediction | Tabular ML | CatBoost | ROC-AUC 0.8548 |

---

# Recommended Projects

## For Computer Vision Roles

- Halo Infinite Object Detection
- Skin Lesion Segmentation
- Simpsons Character Classification

## For NLP / LLM / RAG Roles

- Tourism RAG Assistant
- StackOverflow Semantic Search
- Math Problem Topic Classification

## For Tabular ML / Analytics Roles

- Customer Churn Prediction
- Game of Thrones Survival Prediction

---

# Projects

---

# 1. Halo Infinite Object Detection

Современный object detection pipeline на PyTorch с исследованием detector architectures, assignment strategies и IoU-based losses.

### Что реализовано

- custom PyTorch detector,
- EfficientNet backbone,
- FPN / PAN neck,
- decoupled detection head,
- TAL label assignment,
- DIoU loss,
- YOLOv8 baseline comparison.

### Основная задача

Построение современного object detection pipeline и исследование detector architectures и assignment strategies.

### Лучший результат

| Model | Metric |
|---|---|
| YOLOv8 | mAP50 = **0.915** |
| YOLOv8 | mAP50-95 = **0.6698** |

### Основные ML-инсайты

- TAL + DIoU дали наиболее заметный прирост качества;
- assignment strategy и loss balancing оказались важнее backbone architecture;
- production-grade training recipes критически важны для detector quality.

### Ключевые технологии

- YOLOv8
- PyTorch
- FPN / PAN
- TAL
- DIoU Loss
- Albumentations

---

# 2. Tourism RAG Assistant

Мультимодальная Retrieval-Augmented Generation система для поиска туристических и исторических объектов России.

### Что реализовано

- очистка и агрегация мультимодальных данных,
- retrieval pipeline,
- ChromaDB vector search,
- ColBERT reranking,
- Mistral answer generation,
- embedding space visualization,
- Streamlit demo.

### Основная задача

Построение end-to-end RAG-системы для туристического поиска по историческим и культурным объектам России.

### Лучший результат

| Metric | Value |
|---|---|
| Answer relevancy | **0.85** |
| Context recall | **1.00** |
| Context precision | **0.66** |

### Основные ML-инсайты

- качественная очистка и агрегация данных критичны для retrieval quality;
- reranking существенно улучшает итоговую релевантность;
- embedding space формирует семантические кластеры объектов.

### Ключевые технологии

- RAG
- ColBERT
- ChromaDB
- Mistral-7B
- Sentence Transformers
- UMAP / PCA

---

# 3. Skin Lesion Segmentation with U-Net and SegNet

Proof-of-concept CV-система для автоматической сегментации кожных поражений на дерматоскопических изображениях.

### Что реализовано

- SegNet и U-Net,
- BCE / Dice / Tversky / Focal Loss,
- Dice и IoU evaluation,
- preprocessing pipeline,
- visualization pipeline,
- experiment comparison.

### Основная задача

Автоматическая сегментация кожных поражений на PH2 Dataset.

### Лучший результат

| Model | Loss | Validation Dice |
|---|---|---|
| U-Net | Tversky Loss | **0.900** |

### Основные ML-инсайты

- U-Net существенно превосходит SegNet благодаря skip-connections;
- Tversky Loss оказался наиболее устойчивым для lesion segmentation;
- BCE Loss хуже оптимизирует форму медицинских масок.

### Ключевые технологии

- U-Net
- SegNet
- Semantic Segmentation
- Medical Imaging
- Dice / IoU
- PyTorch

---

# 4. Math Problem Topic Classification with RuBERT and MathBERT

Transformer-based NLP-проект по классификации математических задач по тематическим разделам.

### Что реализовано

- TF-IDF + Logistic Regression baseline,
- TF-IDF + LinearSVC baseline,
- transformer classification pipeline,
- RuBERT и MathBERT experiments,
- frozen / unfrozen backbone comparison,
- mixed precision training,
- attention visualization,
- confusion matrix analysis,
- error analysis.

### Основная задача

Классификация математических задач по тематическим разделам с использованием classical NLP и transformer-based моделей.

### Лучший результат

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| RuBERT tiny2 (unfrozen backbone) | **0.6206** | **0.4758** | **0.5822** |

### Основные ML-инсайты

- TF-IDF baseline оказался неожиданно сильным;
- fine-tuned RuBERT улучшил accuracy, но не решил проблему class imbalance;
- geometry распознаётся лучше всего благодаря устойчивым тематическим маркерам;
- MathBERT не дал явного преимущества над RuBERT на данном датасете.

### Ключевые технологии

- HuggingFace Transformers
- RuBERT
- MathBERT
- PyTorch
- TF-IDF
- LinearSVC
- Attention Visualization

---

# 5. StackOverflow Semantic Search with Word Embeddings

Semantic retrieval pipeline для поиска дубликатов вопросов StackOverflow.

### Что реализовано

- technical tokenizer,
- Word2Vec sentence embeddings,
- cosine similarity ranking,
- Hits@K и DCG@K evaluation,
- retrieval experiments,
- custom Word2Vec training.

### Основная задача

Построение semantic search baseline для поиска похожих технических вопросов.

### Лучший результат

| Metric | Value |
|---|---|
| Hits@10 | **0.641** |
| DCG@10 | **0.516** |

### Основные ML-инсайты

- quality tokenization оказалась важнее переобучения Word2Vec;
- technical tokenizer существенно улучшил retrieval quality;
- mean Word2Vec embeddings дают сильный интерпретируемый baseline.

### Ключевые технологии

- Word2Vec
- Semantic Search
- Retrieval
- Gensim
- NLP
- Ranking Metrics

---

# 6. NLP Text Classification Research with RNN, GRU and LSTM

Исследовательский NLP-проект по сравнению recurrent architectures.

### Что реализовано

- text preprocessing,
- tokenization,
- recurrent architectures,
- stacked RNN experiments,
- pooling strategy comparison,
- hidden dimension analysis,
- gradient clipping stabilization.

### Основная задача

Сравнение recurrent neural architectures для multiclass text classification на AG News.

### Лучший результат

| Model | Accuracy |
|---|---|
| GRU-1x256 + max pooling | **0.9176** |

### Основные ML-инсайты

- GRU показала лучший баланс stability / convergence / accuracy;
- max pooling чаще давал лучшие peak metrics;
- глубокие recurrent stacks ухудшали стабильность.

### Ключевые технологии

- RNN
- GRU
- LSTM
- PyTorch
- NLP
- HuggingFace Datasets

---

# 7. Text-Guided Face Editing with StyleGAN2, CLIP and ArcFace

Generative AI / Computer Vision проект по controllable face editing.

### Что реализовано

- StyleGAN2 generation,
- CLIP-guided optimization,
- ArcFace identity preservation,
- latent optimization,
- ID Loss,
- grid search коэффициентов.

### Основная задача

Редактирование синтетических лиц по текстовому промпту при сохранении identity.

### Лучший результат

| Task | Result |
|---|---|
| Text-guided editing | Stable controllable editing with identity preservation |

### Основные ML-инсайты

- высокий ID Loss лучше сохраняет лицо, но ослабляет редактирование;
- низкая регуляризация усиливает эффект, но увеличивает артефакты;
- баланс между CLIP guidance и identity preservation критичен.

### Ключевые технологии

- StyleGAN2
- OpenAI CLIP
- ArcFace
- Latent Optimization
- Generative AI
- PyTorch

---

# 8. Simpsons Character Classification

Классификация персонажей Simpsons с помощью CNN и transfer learning.

### Что реализовано

- custom CNN architecture,
- transfer learning,
- EfficientNet fine-tuning,
- inference pipeline,
- augmentation pipeline,
- visualization tools.

### Основная задача

Построение multiclass image classification pipeline.

### Лучший результат

| Metric | Value |
|---|---|
| Validation Accuracy | **0.9631** |
| Kaggle Score | **0.98618** |

### Основные ML-инсайты

- transfer learning существенно ускоряет convergence;
- EfficientNet-B0 дал лучший accuracy/efficiency tradeoff;
- fine-tuning pretrained моделей значительно улучшил качество.

### Ключевые технологии

- EfficientNet
- ResNet
- CNN
- Transfer Learning
- PyTorch

---

# 9. Customer Churn Prediction

ML-проект по предсказанию оттока клиентов.

### Что реализовано

- preprocessing pipeline,
- baseline linear models,
- CatBoost training,
- cross-validation,
- hyperparameter tuning,
- ROC-AUC evaluation.

### Основная задача

Сравнение линейных моделей и boosting-подходов для churn prediction.

### Лучший результат

| Metric | Value |
|---|---|
| Validation ROC-AUC | **0.8548** |

### Основные ML-инсайты

- более сложная модель не всегда дает лучшее качество;
- линейные модели могут outperform boosting;
- preprocessing и feature handling часто важнее complexity модели.

### Ключевые технологии

- CatBoost
- Logistic Regression
- ROC-AUC
- Cross-validation
- Business Analytics

---

# 10. Game of Thrones Survival Prediction

Tabular ML-проект по предсказанию выживания персонажей Game of Thrones.

### Что реализовано

- EDA,
- feature engineering,
- обработка high-cardinality категорий,
- OneHot Encoding,
- GridSearchCV,
- model comparison.

### Основная задача

Построение ML pipeline для бинарной классификации на табличных данных.

### Лучший результат

| Metric | Value |
|---|---|
| Holdout Accuracy | **0.8558** |

### Основные ML-инсайты

- feature engineering критически влияет на качество tabular ML;
- distance-based методы могут outperform сложные модели;
- обработка категориальных признаков существенно повышает accuracy.

### Ключевые технологии

- scikit-learn
- Feature Engineering
- GridSearchCV
- OneHotEncoding
- Tabular ML

---

# What These Projects Demonstrate

## Computer Vision

- image classification,
- object detection,
- semantic segmentation,
- generative image editing,
- medical image analysis.

## NLP / LLM / Retrieval

- semantic search,
- text classification,
- retrieval pipelines,
- reranking,
- vector databases,
- transformer fine-tuning,
- RAG systems,
- embedding analysis.

## Machine Learning

- supervised learning,
- boosting,
- cross-validation,
- hyperparameter tuning,
- feature engineering,
- reproducible ML pipelines.

## Engineering

- modular project structure,
- experiment management,
- evaluation pipelines,
- visualization tools,
- research-oriented experimentation.

---

# Tech Stack

## ML / DL

- Python
- PyTorch
- scikit-learn
- CatBoost
- Ultralytics YOLO
- HuggingFace Transformers

## NLP / Retrieval

- Sentence Transformers
- ChromaDB
- ColBERT
- Gensim
- OpenAI CLIP

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

---

# Repository Structure

```text
ML_Portfolio/
│
├── Halo_Object_Detection/
├── Tourism_RAG_Assistant/
├── Skin_Lesion_Segmentation/
├── Math_Problem_Classification/
├── StackOverflow_Semantic_Search/
├── NLP_Text_Classification/
├── Text_Guided_Face_Editing/
├── Simpsons_Classification/
├── Customer_Churn_Prediction/
├── Game_of_Thrones_Survival/
│
├── requirements.txt
└── README.md
```

---

# Main Repository Goals

Репозиторий демонстрирует:

- end-to-end ML pipeline development;
- reproducible experimentation;
- model evaluation and error analysis;
- retrieval and generative AI systems;
- CV and NLP workflows;
- research-oriented ML engineering approach.

---

# Possible Future Improvements

- MLflow / experiment tracking,
- Dockerization,
- CI/CD,
- Optuna hyperparameter optimization,
- SHAP explainability,
- deployment inference services,
- FastAPI / Streamlit demos,
- distributed training.

---

# GitHub

GitHub Profile: https://github.com/MataNerdy