from __future__ import annotations

import argparse

from src.rag import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 RAG 服务进行一次问答示例")
    parser.add_argument(
        "question",
        nargs="?",
        default="Jimmer 是什么？",
        help="想要查询的问题。如果未提供，将使用默认示例问题。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="可选：覆盖默认的检索文档数量。",
    )
    args = parser.parse_args()

    service = RAGService(top_k=args.top_k)
    result = service.ask(args.question)

    print("=== 回答 ===")
    print(result.answer.strip())
    if result.sources:
        print("\n=== 引用来源 ===")
        for source in result.sources:
            index = source.get("index")
            label = source.get("source")
            title = source.get("title")
            prefix = f"[{index}] " if index is not None else ""
            suffix = f" ｜ 标题: {title}" if title else ""
            print(f"{prefix}{label}{suffix}")


if __name__ == "__main__":
    main()