"""文档灌入脚本。

将 data/ 目录下的所有支持文档加载、切片后存入向量库。
用法: uv run python ingest.py [--data-dir DATA_DIR] [--reset]
"""

import argparse
import sys

from src.chunker import split_documents
from src.document_loader import load_directory
from src.vector_store import VectorStore


def main() -> None:
    """执行文档灌入流程。"""
    parser = argparse.ArgumentParser(description="文档灌入向量库")
    parser.add_argument("--data-dir", default="data", help="文档目录路径")
    parser.add_argument("--reset", action="store_true", help="灌入前清空向量库")
    args = parser.parse_args()

    # 1. 加载文档
    print(f"📂 加载文档: {args.data_dir}")
    documents = load_directory(args.data_dir)
    if not documents:
        print("⚠️  未找到支持的文档文件")
        sys.exit(1)
    print(f"   找到 {len(documents)} 个文档片段")
    sources = {doc.metadata.get("source", "unknown") for doc in documents}
    for source in sorted(sources):
        count = sum(1 for d in documents if d.metadata.get("source") == source)
        print(f"   - {source} ({count} 页/段)")

    # 2. 文本切片
    print("\n✂️  文本切片 (chunk_size=500, overlap=100)")
    chunks = split_documents(documents, chunk_size=500, chunk_overlap=100)
    print(f"   产生 {len(chunks)} 个切片")

    # 3. 写入向量库
    store = VectorStore()
    if args.reset:
        print("\n🗑️  清空向量库")
        store.delete_collection()

    print("\n📥 写入向量库")
    count = store.add_documents(chunks)
    print(f"   写入 {count} 条记录，总计 {store.count} 条")

    # 4. 验证
    print("\n🔍 验证检索")
    results = store.search("什么是装饰器", top_k=3)
    for i, r in enumerate(results):
        print(f"   [{i+1}] {r['source']} (score: {r['score']:.4f})")
        print(f"       {r['content'][:80]}...")

    print("\n✅ 灌入完成")


if __name__ == "__main__":
    main()
