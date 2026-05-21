import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import CONFIG


def row_to_document(row: pd.Series) -> Document:
    page_content = "\n".join([
        f"Название: {row['Name']}",
        f"Город: {row['City']}",
        "",
        "Описание (WikiData):",
        str(row["description"]),
        "",
        "Описание по фото:",
        str(row["text_final_clean"]),
    ])
    metadata = {
        "place_hints": row["place_hints"],
        "wikidata_id": row["WikiData"],
        "city": row["City"],
        "image_base64": row.get("image", ""),
    }
    return Document(page_content=page_content, metadata=metadata)


def build_vector_store(df: pd.DataFrame) -> Chroma:
    embedding_model = HuggingFaceEmbeddings(
        model_name=CONFIG.embedding_model,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    documents = [row_to_document(row) for _, row in df.iterrows()]
    return Chroma.from_documents(
        documents,
        embedding_model,
        collection_name=CONFIG.collection_name,
        persist_directory=str(CONFIG.chroma_dir),
    )


def load_vector_store() -> Chroma:
    embedding_model = HuggingFaceEmbeddings(
        model_name=CONFIG.embedding_model,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=CONFIG.collection_name,
        persist_directory=str(CONFIG.chroma_dir),
        embedding_function=embedding_model,
    )


if __name__ == "__main__":
    df = pd.read_csv(CONFIG.processed_data_path)
    build_vector_store(df)
    print(f"Vector store saved to {CONFIG.chroma_dir}")
