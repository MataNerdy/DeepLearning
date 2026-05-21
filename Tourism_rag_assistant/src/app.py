import streamlit as st
import pandas as pd

from config import CONFIG
from rag_pipeline import answer_with_tourism_rag, build_reader_llm, load_reranker
from vector_store import load_vector_store


@st.cache_resource
def load_resources():
    knowledge_index = load_vector_store()
    reranker = load_reranker()
    llm, prompt_template = build_reader_llm()
    return knowledge_index, reranker, llm, prompt_template


st.set_page_config(page_title="Tourism RAG Assistant", layout="wide")
st.title("Tourism RAG Assistant")
st.caption("RAG-гид по достопримечательностям: E5 embeddings + Chroma + ColBERT reranking + LLM reader")

question = st.text_input(
    "Вопрос",
    value="Что посмотреть в Ярославле, если люблю старинные храмы и набережную?",
)

num_docs = st.slider("Сколько документов оставить после reranking", 1, 8, 5)

if st.button("Спросить RAG"):
    knowledge_index, reranker, llm, prompt_template = load_resources()
    answer, docs = answer_with_tourism_rag(
        question=question,
        llm=llm,
        prompt_template=prompt_template,
        knowledge_index=knowledge_index,
        reranker=reranker,
        num_retrieved_docs=20,
        num_docs_final=num_docs,
    )
    st.subheader("Ответ")
    st.write(answer)

    st.subheader("Использованные документы")
    for i, doc in enumerate(docs, start=1):
        with st.expander(f"Документ {i}"):
            st.text(doc)
