from pathlib import Path
import os


class config:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    SRC_DATASET_DIR = ROOT_DIR / "train" / "dataset" / "src"
    INDEX_DIR = ROOT_DIR / "src" / "faiss_index"
    INDEX_NAME = "index"

    REPO_URL = "https://github.com/babyfish-ct/jimmer-doc.git"
    BRANCH = "master"
    DEPTH = 1

    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHAT_MODEL = (
        os.getenv("CHAT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("DASHSCOPE_MODEL")
        or "gpt-4o"
    )
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 256))
    DEFAULT_TOP_K = 80
    ADAPTIVE_RETRIEVAL = os.getenv("ADAPTIVE_RETRIEVAL", "false").lower() in {"1", "true", "yes", "on"}
    RETRIEVER_SEARCH_TYPE = os.getenv("RETRIEVER_SEARCH_TYPE", "mmr")
    USE_BM25 = os.getenv("USE_BM25", "true").lower() in {"1", "true", "yes", "on"}
    VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.6"))
    BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
