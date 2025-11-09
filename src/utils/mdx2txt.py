from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Tuple

__all__ = ["parse_mdx_file"]


def parse_mdx_file(file_path: str) -> Tuple[dict, str]:
    """调用 Node.js 脚本解析 MDX 文件，返回元数据和纯文本内容。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")

    node_dir = Path(__file__).resolve().parent.parent / "node"
    script_path = node_dir / "parse_mdx.js"

    if not script_path.exists():
        raise FileNotFoundError(f"缺少 MDX 解析脚本：{script_path}")

    result = subprocess.run(
        ["node", str(script_path), file_path],
        capture_output=True,
        text=True,
        cwd=str(node_dir),
        timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(f"MDX 解析失败（{file_path}）：{result.stderr}")

    parsed_result = json.loads(result.stdout)
    metadata = parsed_result.get("metadata", {})
    content = parsed_result.get("content", "")

    return metadata, content


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="解析 MDX 文件为纯文本")
    parser.add_argument("file", help="待解析的 MDX 文件路径")
    args = parser.parse_args()

    meta, text = parse_mdx_file(args.file)
    print(json.dumps({"metadata": meta, "content": text}, ensure_ascii=False, indent=2))