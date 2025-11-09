from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.mdx2txt import parse_mdx_file

logger = logging.getLogger(__name__)

MARKUP_SUFFIXES = {".md", ".mdx", ".markdown"}
CODE_SUFFIXES = {".py", ".js", ".ts", ".tsx"}
SUPPORTED_SUFFIXES = MARKUP_SUFFIXES | CODE_SUFFIXES


def _strip_front_matter(text: str) -> str:
    """Remove YAML front matter blocks commonly used in markdown/mdx files."""
    if text.lstrip().startswith("---"):
        return re.sub(r"^---.*?---\s*", "", text, flags=re.DOTALL)
    return text


def _normalise_markdown(text: str) -> str:
    """Reduce MD/MDX noise such as imports or JSX snippets."""
    text = _strip_front_matter(text)

    cleaned_lines: List[str] = []
    skip_prefixes = ("import ", "export ", "const ", "let ", "var ", "type ", "interface ")

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(skip_prefixes) and stripped.endswith(";"):
            continue
        cleaned_lines.append(line)

    normalised = "\n".join(cleaned_lines)
    # Collapse multiple blank lines to make chunking more efficient
    normalised = re.sub(r"\n{3,}", "\n\n", normalised)
    return normalised.strip()


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as file:
        return file.read()


def _normalise_mdx(path: Path, fallback_text: str) -> tuple[str, dict]:
    try:
        mdx_metadata, mdx_content = parse_mdx_file(path.as_posix())
        if mdx_content and mdx_content.strip():
            return mdx_content, mdx_metadata or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("MDX parse failed for %s: %s", path, exc)
    return fallback_text, {}


def _create_document(path: Path, root_dir: Path) -> Document | None:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return None

    raw_text = _read_text(path)
    extra_metadata: dict = {}

    if suffix == ".mdx":
        raw_text, extra_metadata = _normalise_mdx(path, raw_text)
        logger.info("Parsed MDX metadata: %s", extra_metadata)

    text = raw_text
    if suffix in MARKUP_SUFFIXES:
        text = _normalise_markdown(text)

    if not text.strip():
        return None

    relative_path = path.relative_to(root_dir)
    metadata = {
        "source": str(relative_path),
        "file_path": str(path),
        "category": str(relative_path.parent) if relative_path.parent != Path(".") else "",
    }

    for key, value in extra_metadata.items():
        if isinstance(value, (str, int, float)):
            metadata[f"mdx_{key}"] = value

    heading_match = re.search(r"^#\s+(.*)$", text, flags=re.MULTILINE)
    if heading_match:
        metadata["title"] = heading_match.group(1).strip()

    return Document(page_content=text, metadata=metadata)

def load_documents(root_dir: Path | str) -> List[Document]:
    """Load supported documents from ``root_dir`` into LangChain Document objects."""
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root documentation directory not found: {root}")

    documents: List[Document] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        document = _create_document(file_path, root)
        if document:
            documents.append(document)
            logger.info("Loaded document: %s", file_path.absolute())

    return documents

def chunk_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Split documents into overlapping chunks suitable for embedding."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    res = splitter.split_documents(list(documents))

    return res

