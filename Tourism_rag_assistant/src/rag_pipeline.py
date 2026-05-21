from typing import Any, List, Optional, Tuple

import torch
from ragatouille import RAGPretrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

from config import CONFIG


SYSTEM_PROMPT = """Ты — аккуратный и дружелюбный туристический гид.
Отвечай ТОЛЬКО на основе предоставленного контекста.
Запрещено придумывать факты, которых нет в тексте.
Если информации недостаточно, честно скажи об этом.
Отвечай по-русски, ясно и по делу, как человеку, который планирует поездку.
Когда уместно, упоминай название достопримечательности и город."""


def build_reader_llm() -> Tuple[Pipeline, str]:
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.reader_model)
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG.reader_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    prompt_template = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Контекст:\n{context}\n---\nВопрос: {question}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    llm = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=400,
        return_full_text=False,
    )
    return llm, prompt_template


def answer_with_tourism_rag(
    question: str,
    llm: Pipeline,
    prompt_template: str,
    knowledge_index: Any,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 20,
    num_docs_final: int = 5,
) -> Tuple[str, List[str]]:
    docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    texts = list(dict.fromkeys([doc.page_content for doc in docs]))

    if reranker is not None and len(texts) > 0:
        reranked = reranker.rerank(question, texts, k=num_docs_final)
        texts = [item["content"] for item in reranked]
    else:
        texts = texts[:num_docs_final]

    context = "\nExtracted documents:\n" + "".join(
        f"Document {i}:::\n{text}\n\n" for i, text in enumerate(texts)
    )
    prompt = prompt_template.format(question=question, context=context)
    generation = llm(prompt, max_new_tokens=120, do_sample=False, temperature=0.0)

    answer = generation[0]["generated_text"] if isinstance(generation, list) else generation["generated_text"]
    return answer, texts


def load_reranker() -> RAGPretrainedModel:
    return RAGPretrainedModel.from_pretrained(CONFIG.reranker_model)
