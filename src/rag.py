from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS

from src.pipeline import build_vectorstore, create_rag_chain, get_retriever


@dataclass
class RAGResult:
    answer: str
    sources: List[dict]


class RAGService:
    """High-level facade that wraps retrieval and generation logic."""

    def __init__(self, *, vectorstore: Optional[FAISS] = None, top_k: Optional[int] = None):
        self.vectorstore = vectorstore or build_vectorstore()
        self.top_k = top_k
        self.retriever = get_retriever(self.vectorstore, search_kwargs={"k": top_k} if top_k else None)
        self.chain = create_rag_chain(self.vectorstore, top_k=top_k)

    def ask(self, question: str) -> RAGResult:
        """Run the full RAG pipeline for a given question."""
        response: Dict[str, Any] = self.chain.invoke(question)
        return RAGResult(answer=response["answer"], sources=response.get("sources", []))

    def retrieve(self, question: str):
        """Expose raw retrieval results for debugging or evaluation."""
        return self.retriever.invoke(question)
