from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    raw_data_url: str = "https://drive.google.com/uc?id=1P1BsvI2jPN3fEqjc2YZxmQ-MTs22WVUk"
    raw_data_path: Path = Path("data/raw/file.csv")
    processed_data_path: Path = Path("data/processed/tourism_rag.csv")
    eval_data_path: Path = Path("reports/df_eval_100.parquet")
    chroma_dir: Path = Path("chroma_db/tourism")
    collection_name: str = "tourism_rag"

    embedding_model: str = "intfloat/multilingual-e5-base"
    cleaning_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "colbert-ir/colbertv2.0"
    reader_model: str = "mistralai/Mistral-7B-Instruct-v0.3"

    target_n_places: int = 250
    similarity_threshold: float = 0.25
    min_caption_len: int = 20
    topk_per_place: int = 3
    max_final_chars: int = 2500
    random_state: int = 42


CONFIG = ProjectConfig()
