from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from src.config import config
from src.parse import chunk_documents, load_documents

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _bm25_preprocess(text: str) -> List[str]:
    """Tokenize text for BM25 retrieval, with naive Chinese handling."""
    tokens: List[str] = []
    buffer: List[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            tokens.append("".join(buffer))
            buffer = []

    for char in text.lower():
        if "\u4e00" <= char <= "\u9fff":
            flush_buffer()
            tokens.append(char)
        elif char.isalnum():
            buffer.append(char)
        else:
            flush_buffer()

    flush_buffer()
    return tokens


class HybridRetriever:
    """Combine vector store and BM25 retrieval results."""

    def __init__(
        self,
        *,
        vector_retriever,
        bm25_retriever: BM25Retriever,
        vector_weight: float,
        bm25_weight: float,
        top_k: int,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = max(vector_weight, 0.0)
        self.bm25_weight = max(bm25_weight, 0.0)
        self.top_k = top_k

    def invoke(self, query: str) -> List[Document]:
        vector_docs = list(self.vector_retriever.invoke(query))
        bm25_docs = list(self.bm25_retriever.invoke(query))

        target_k = self.top_k if self.top_k > 0 else max(len(vector_docs), len(bm25_docs))
        if target_k <= 0:
            return []

        total_weight = self.vector_weight + self.bm25_weight
        if total_weight <= 0:
            vector_quota = target_k
        else:
            vector_quota = int(round(self.vector_weight / total_weight * target_k))
        vector_quota = min(max(vector_quota, 0), target_k)
        bm25_quota = target_k - vector_quota

        combined: List[Document] = []
        seen_sources = set()

        def add_from_list(docs: Sequence[Document], limit: int | None = None) -> None:
            count = 0
            for doc in docs:
                source_key = doc.metadata.get("source") or doc.metadata.get("file_path")
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)
                combined.append(doc)
                count += 1
                if limit is not None and count >= limit:
                    break
                if len(combined) >= target_k:
                    break

        add_from_list(vector_docs, vector_quota if vector_quota > 0 else None)
        if len(combined) < target_k and bm25_quota > 0:
            add_from_list(bm25_docs, bm25_quota)

        if len(combined) < target_k:
            add_from_list(vector_docs)
        if len(combined) < target_k:
            add_from_list(bm25_docs)

        return combined[:target_k]

    def batch(self, queries: Sequence[str]) -> List[List[Document]]:
        return [self.invoke(query) for query in queries]


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


@lru_cache(maxsize=4)
def _load_chunks_cached(chunk_size: int, chunk_overlap: int) -> tuple[Document, ...]:
    logger.info("Loading documents from %s", config.SRC_DATASET_DIR)
    documents = load_documents(config.SRC_DATASET_DIR)
    logger.info("Loaded %d documents, splitting into chunks.", len(documents))

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    logger.info("Generated %d chunks for embedding.", len(chunks))
    return tuple(chunks)


def build_vectorstore(*, force_rebuild: bool = False) -> FAISS:
    """
    Build (or load) the FAISS vector store from the documentation corpus.

    Args:
        force_rebuild: When True, reprocess the corpus even if a cached index exists.
    """
    logger.info("Building FAISS index for %s", config.SRC_DATASET_DIR)
    index_dir = Path(config.INDEX_DIR).expanduser().resolve()
    _ensure_directory(index_dir)

    embeddings = _get_embeddings()

    if not force_rebuild:
        try:
            return FAISS.load_local(
                folder_path=str(index_dir),
                embeddings=embeddings,
                index_name=config.INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
        except (FileNotFoundError, ValueError):
            logger.info("Existing FAISS index not found, rebuilding from source documents.")

    if force_rebuild:
        _load_chunks_cached.cache_clear()

    chunks = list(_load_chunks_cached(config.CHUNK_SIZE, config.CHUNK_OVERLAP))

    vectorstore = FAISS.from_documents(chunks, embeddings)

    logger.info(f"Vectorstore built with {vectorstore.index.ntotal} vectors")

    vectorstore.save_local(
        folder_path=str(index_dir),
        index_name=config.INDEX_NAME,
    )
    logger.info("Saved FAISS index to %s", index_dir)
    return vectorstore


def get_retriever(
    vectorstore: FAISS | None = None,
    *,
    search_kwargs: dict | None = None,
):
    if vectorstore is None:
        vectorstore = build_vectorstore()

    kwargs = {"k": config.DEFAULT_TOP_K}
    if search_kwargs:
        kwargs.update(search_kwargs)

    vector_retriever = vectorstore.as_retriever(
        search_type=config.RETRIEVER_SEARCH_TYPE,
        search_kwargs=kwargs,
    )

    if not config.USE_BM25:
        return vector_retriever

    chunks = list(_load_chunks_cached(config.CHUNK_SIZE, config.CHUNK_OVERLAP))
    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        preprocess_func=_bm25_preprocess,
    )
    bm25_retriever.k = kwargs.get("k", config.DEFAULT_TOP_K)

    logger.info(
        "Using hybrid retriever (vector weight %.2f, bm25 weight %.2f)",
        config.VECTOR_WEIGHT,
        config.BM25_WEIGHT,
    )

    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        vector_weight=config.VECTOR_WEIGHT,
        bm25_weight=config.BM25_WEIGHT,
        top_k=kwargs.get("k", config.DEFAULT_TOP_K),
    )


def format_docs(docs: Sequence[Document]) -> str:
    formatted_sections: List[str] = []
    logger.info(f"formatting docs size: {len(docs)}")
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source") or doc.metadata.get("file_path") or "unknown"
        title = doc.metadata.get("title")
        header = f"[{idx}] 来源: {source}"
        if title:
            header += f" ｜ 标题: {title}"
        formatted_sections.append(f"{header}\n{doc.page_content.strip()}")
        logger.info(f"formated: {idx}. {header}")
    return "\n\n".join(formatted_sections)


def create_rag_chain(
    vectorstore: FAISS | None = None,
    *,
    top_k: int | None = None,
) -> RunnableSerializable:
    search_kwargs = {"k": top_k} if top_k else None
    retriever = get_retriever(vectorstore, search_kwargs=search_kwargs)

    prompt = PromptTemplate.from_template(
        """你是一个 Jimmer 框架专家。请基于提供的上下文准确回答用户的问题，并在回答末尾列出引用的来源编号。
上下文：
{context}

问题：{question}

请用中文回答。如果上下文不足以回答，请明确说明。"""
    )

    chat_kwargs = {"model": config.CHAT_MODEL, "temperature": 0}
    if config.OPENAI_BASE_URL:
        chat_kwargs["base_url"] = config.OPENAI_BASE_URL
    if config.OPENAI_API_KEY:
        chat_kwargs["api_key"] = config.OPENAI_API_KEY

    llm = ChatOpenAI(**chat_kwargs)

    def _retrieve(question: str):
        docs = retriever.invoke(question)
        logger.info(f"retrieved docs size: {len(docs)}")
        return {"question": question, "docs": docs}

    def _prepare(inputs: dict) -> dict:
        docs = inputs["docs"]
        deduped_docs: List[Document] = []
        seen_sources = set()
        for doc in docs:
            source_key = doc.metadata.get("source") or doc.metadata.get("file_path")
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            deduped_docs.append(doc)

        return {
            "question": inputs["question"],
            "context": format_docs(deduped_docs),
            "sources": [
                {
                    "index": idx + 1,
                    "source": doc.metadata.get("source") or doc.metadata.get("file_path"),
                    "title": doc.metadata.get("title"),
                }
                for idx, doc in enumerate(deduped_docs)
            ],
        }

    chain: RunnableSerializable = (
        RunnableLambda(_retrieve)
        | RunnableLambda(_prepare)
        | (
            {
                "answer": prompt | llm | StrOutputParser(),
                "sources": RunnablePassthrough() | RunnableLambda(lambda x: x["sources"]),
            }
        )
    )

    return chain
