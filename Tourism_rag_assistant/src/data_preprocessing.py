import base64
import re
from io import BytesIO
from pathlib import Path
from typing import Iterable, List

import gdown
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import CONFIG


SUSPECT_PATTERNS = [
    r"\bportrait(s)?\b", r"\bmodels?\b", r"\bpeople\b",
    r"\bpark(?:ing|ed)\b", r"\ban?\s+(?:view|picture|image|photo)\s+of\b",
    r"\bpainting\b", r"\bpicture of\b", r"\ban?\s+image of\b",
    r"\bcars?\b", r"\bmaps?\b", r"\badvert(?:ising|isement)?\b",
    r"\bмем(?!ориал)\b", r"\bmemes?\b", r"\bбаннер(?:ы)?\b",
    r"\bposter\b", r"\bbillboards?\b", r"\binstagram\b",
    r"\bvk\.com\b", r"\btiktok\b",
]


def download_raw_data(output_path: Path = CONFIG.raw_data_path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path
    gdown.download(CONFIG.raw_data_url, str(output_path), quiet=False)
    return output_path


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u200b", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()


def make_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["description", "en_txt", "Name", "City"]:
        df[col] = df[col].map(normalize_text)

    df["place_hints"] = (df["Name"] + " " + df["City"]).map(normalize_text)
    df["text"] = (
        df["Name"] + " " + df["City"] + " " + df["description"] + " " + df["en_txt"]
    ).map(normalize_text)
    return df


def pick_best_name(variants: Iterable[str]) -> str:
    uniq = list(set(variants))
    return max(uniq, key=len)


def flag_suspect_text(text: str) -> bool:
    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in SUSPECT_PATTERNS]
    text = (text or "").lower()
    return any(pattern.search(text) for pattern in compiled)


def topk_typical_texts(texts: List[str], k: int = 3) -> List[int]:
    if len(texts) <= k:
        return list(range(len(texts)))

    vectorizer = TfidfVectorizer(
        max_features=30000,
        token_pattern=r"(?u)\b[\w\-]{2,}\b",
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(texts)
    centroid = np.asarray(matrix.mean(axis=0)).ravel().reshape(1, -1)
    sims = cosine_similarity(matrix, centroid).ravel()

    picked = []
    used = set()
    for idx in np.argsort(-sims):
        if texts[idx] in used:
            continue
        picked.append(idx)
        used.add(texts[idx])
        if len(picked) >= k:
            break
    return picked


def clean_caption(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"\barafed\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def build_text_final(group: pd.DataFrame, k: int = 3, max_chars: int = 2500) -> str:
    texts = group["en_txt"].fillna("").astype(str).tolist()
    idxs = topk_typical_texts(texts, k=k)
    selected = [texts[i] for i in idxs if texts[i].strip()]
    final = " ".join(selected)
    return clean_caption(final[:max_chars])


def balanced_sample_by_city(
    df: pd.DataFrame,
    target_n: int = CONFIG.target_n_places,
    random_state: int = CONFIG.random_state,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    groups = list(df.groupby("City", dropna=False))
    base_quota = max(1, target_n // max(1, len(groups)))

    picked_idx = []
    for _, group in groups:
        n = min(base_quota, len(group))
        picked_idx.extend(group.sample(n=n, random_state=int(rng.integers(1e9))).index)

    if len(picked_idx) < target_n:
        remaining = df.drop(index=picked_idx).sort_values(
            ["n_records_used", "n_records_total"], ascending=False
        )
        picked_idx.extend(remaining.head(target_n - len(picked_idx)).index)

    return df.loc[picked_idx].head(target_n).reset_index(drop=True)


def prepare_dataset(
    raw_path: Path = CONFIG.raw_data_path,
    output_path: Path = CONFIG.processed_data_path,
) -> pd.DataFrame:
    if not raw_path.exists():
        download_raw_data(raw_path)

    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    df = raw.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.dropna(subset=["description", "WikiData"]).reset_index(drop=True)
    df = make_text_columns(df)

    # Remove exact duplicate generated captions.
    df = df.drop_duplicates(subset=["en_txt"]).reset_index(drop=True)

    # Canonicalize alternative names mapped to the same WikiData ID.
    canonical_name = (
        df.groupby("WikiData")["place_hints"]
        .apply(pick_best_name)
        .reset_index(name="canonical_place_name")
    )
    df = df.merge(canonical_name, on="WikiData", how="right")
    df["place_hints"] = df["canonical_place_name"]

    # Regex filter for obvious non-tourism / non-landmark artifacts.
    df["is_suspect_regex"] = df["text"].apply(flag_suspect_text)
    df = df[~df["is_suspect_regex"]].copy()

    # Semantic consistency check between WikiData description and generated image caption.
    model = SentenceTransformer(CONFIG.cleaning_model)
    desc_emb = model.encode(
        df["description"].tolist(),
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    img_emb = model.encode(
        df["en_txt"].tolist(),
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    df["similarity"] = (desc_emb * img_emb).sum(axis=1)

    mask_good = (
        (df["en_txt"].str.len() >= CONFIG.min_caption_len)
        & (df["similarity"] >= CONFIG.similarity_threshold)
    )
    df_good = df[mask_good].copy()

    rows = []
    for place, group_all in df.groupby("place_hints", dropna=False):
        group_good = df_good.loc[group_all.index.intersection(df_good.index)]
        use_group = group_good if len(group_good) > 0 else group_all

        row0 = group_all.iloc[0]
        rows.append({
            "place_hints": place,
            "WikiData": row0.get("WikiData", ""),
            "Name": row0.get("Name", ""),
            "City": row0.get("City", ""),
            "Lon": row0.get("Lon", np.nan),
            "Lat": row0.get("Lat", np.nan),
            "description": row0.get("description", ""),
            "image": row0.get("image", ""),
            "text_final_clean": build_text_final(
                use_group,
                k=CONFIG.topk_per_place,
                max_chars=CONFIG.max_final_chars,
            ),
            "n_records_total": len(group_all),
            "n_records_used": len(use_group),
            "sim_mean_used": float(use_group["similarity"].mean()),
            "sim_min_used": float(use_group["similarity"].min()),
        })

    df_places = pd.DataFrame(rows)
    df_rag = balanced_sample_by_city(df_places)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_rag.to_csv(output_path, index=False)
    return df_rag


def decode_image(image_base64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")


if __name__ == "__main__":
    df_rag = prepare_dataset()
    print(f"Prepared {len(df_rag)} RAG documents")
