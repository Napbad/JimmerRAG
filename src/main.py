from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clone_repo import clone_git_repository
from src.config import config
from src.pipeline import build_vectorstore, create_rag_chain


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jimmer RAG pipeline CLI")
    parser.add_argument(
        "--question",
        type=str,
        help="要查询的问题。如果未提供，则仅构建或加载向量索引。",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="忽略已有索引，重新生成 FAISS 向量库。",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="跳过文档仓库克隆步骤（默认当目录为空时会自动克隆）。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.DEFAULT_TOP_K,
        help=f"检索返回的文档数量（默认 {config.DEFAULT_TOP_K}）。",
    )
    return parser.parse_args(argv)


def ensure_dataset(skip_clone: bool) -> None:
    dataset_path = Path(config.SRC_DATASET_DIR)
    if skip_clone:
        return

    if dataset_path.exists() and any(dataset_path.iterdir()):
        return

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    cloned = clone_git_repository(
        config.REPO_URL,
        dataset_path,
        branch=config.BRANCH,
        depth=config.DEPTH,
    )
    if not cloned:
        raise RuntimeError(
            f"无法克隆文档仓库到 {dataset_path}，请检查网络或手动准备数据集。"
        )


def run_cli(argv: Optional[list[str]] = None) -> None:
    logging.info("Jimmer RAG pipeline CLI")
    args = parse_args(argv)
    ensure_dataset(args.skip_clone)

    vectorstore = build_vectorstore(force_rebuild=args.rebuild_index)
    rag_chain = create_rag_chain(vectorstore, top_k=args.top_k)

    if not args.question:
        print("向量索引已准备就绪。使用 `--question \"...\"` 进行查询。")
        return

    result = rag_chain.invoke(args.question)

    answer = result["answer"]
    sources = result.get("sources", [])

    print("\n=== 回答 ===")
    print(answer.strip())
    if sources:
        print("\n=== 引用来源 ===")
        for source in sources:
            index = source.get("index")
            label = source.get("source", "未知来源")
            title = source.get("title")
            suffix = f" ｜ 标题: {title}" if title else ""
            prefix = f"[{index}] " if index is not None else ""
            print(f"{prefix}{label}{suffix}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_cli(sys.argv[1:])