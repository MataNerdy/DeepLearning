import json
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from config import CONFIG
from rag_pipeline import answer_with_tourism_rag


def generate_followup_questions(answer: str, llm, n: int = 3) -> List[str]:
    prompt = f"""
Ты ассистент, который придумывает уточняющие вопросы по тексту.
Сформулируй ровно {n} вопросов.

Формат ответа строго JSON:
{{"questions": ["...", "...", "..."]}}

Ответ:
\"\"\"{answer}\"\"\"
""".strip()
    out = llm(prompt, max_new_tokens=200, do_sample=True, temperature=0.6, top_p=0.9)
    text = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]

    try:
        json_str = re.search(r"\{.*\}", text, re.DOTALL).group(0)
        questions = json.loads(json_str).get("questions", [])
    except Exception:
        questions = []

    if not questions:
        questions = [
            "О чём говорится в тексте?",
            "Где находится описанный объект?",
            "Чем он может быть интересен туристу?",
        ]
    return questions[:n]


def build_eval_dataset(
    df: pd.DataFrame,
    llm,
    prompt_template: str,
    knowledge_index,
    n_samples: int = 100,
    save_path: str = str(CONFIG.eval_data_path),
) -> pd.DataFrame:
    df_eval = df.sample(n_samples, random_state=CONFIG.random_state).copy()
    df_eval["question"] = df_eval.apply(
        lambda row: f"Что такое {row['Name']} в городе {row['City']}?",
        axis=1,
    )

    if os.path.exists(save_path):
        out = pd.read_parquet(save_path)
    else:
        out = pd.DataFrame()

    for _, row in tqdm(df_eval.iloc[len(out):].iterrows(), total=n_samples - len(out)):
        answer, contexts = answer_with_tourism_rag(
            question=row["question"],
            llm=llm,
            prompt_template=prompt_template,
            knowledge_index=knowledge_index,
            num_retrieved_docs=5,
            num_docs_final=3,
        )
        out_row = row.to_dict()
        out_row["answer"] = answer
        out_row["contexts"] = contexts
        out_row["ground_truths"] = [row["description"]]
        out_row["followup_questions"] = generate_followup_questions(answer, llm)
        out = pd.concat([out, pd.DataFrame([out_row])], ignore_index=True)
        out.to_parquet(save_path, index=False)

    return out


def answer_relevancy(
    df_eval: pd.DataFrame,
    model: SentenceTransformer | None = None,
    model_name: str = CONFIG.embedding_model,
) -> Tuple[float, List[float]]:
    if model is None:
        model = SentenceTransformer(model_name)

    answers = df_eval["answer"].astype(str).tolist()
    followups = df_eval["followup_questions"].tolist()
    answer_embeddings = model.encode(answers, normalize_embeddings=True)

    scores = []
    for i, questions in enumerate(followups):
        if not isinstance(questions, list) or len(questions) == 0:
            scores.append(0.0)
            continue
        question_embeddings = model.encode(questions, normalize_embeddings=True)
        sims = question_embeddings @ answer_embeddings[i]
        scores.append(float(np.mean(sims)))
    return float(np.mean(scores)), scores


def is_relevant_context(row: pd.Series, context: str) -> bool:
    name_tokens = set(str(row["Name"]).lower().split())
    city_tokens = set(str(row["City"]).lower().split())
    context_tokens = set(str(context).lower().split())
    return bool(name_tokens & context_tokens) and bool(city_tokens & context_tokens)


def context_recall(df_eval: pd.DataFrame) -> Tuple[float, List[float]]:
    scores = []
    for _, row in df_eval.iterrows():
        contexts = row["contexts"]
        flags = [is_relevant_context(row, ctx) for ctx in contexts] if isinstance(contexts, list) else []
        scores.append(1.0 if any(flags) else 0.0)
    return float(np.mean(scores)), scores


def context_precision(df_eval: pd.DataFrame) -> Tuple[float, List[float]]:
    scores = []
    for _, row in df_eval.iterrows():
        contexts = row["contexts"]
        if not isinstance(contexts, list) or len(contexts) == 0:
            scores.append(0.0)
            continue
        flags = [is_relevant_context(row, ctx) for ctx in contexts]
        scores.append(float(np.mean(flags)))
    return float(np.mean(scores)), scores
