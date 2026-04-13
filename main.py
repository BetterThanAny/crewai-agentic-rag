"""CLI 入口 — 支持单次查询和交互式多轮对话。

用法:
    单次查询:  uv run python main.py --query "什么是装饰器？"
    交互模式:  uv run python main.py
    详细模式:  uv run python main.py --verbose
    灌入文档:  uv run python main.py --ingest [--data-dir data] [--reset]
"""

from __future__ import annotations

import argparse
import sys

from src.crew import RAGCrew


def _run_ingest(data_dir: str, reset: bool) -> None:
    """执行文档灌入流程。

    Args:
        data_dir: 文档目录路径。
        reset: 是否清空向量库后重新灌入。
    """
    from src.chunker import split_documents
    from src.document_loader import load_directory
    from src.vector_store import VectorStore

    print(f"📂 加载文档: {data_dir}")
    documents = load_directory(data_dir)
    if not documents:
        print("⚠️  未找到支持的文档文件")
        sys.exit(1)

    sources = {doc.metadata.get("source", "unknown") for doc in documents}
    print(f"   找到 {len(documents)} 个文档片段，来自 {len(sources)} 个文件")

    print("✂️  文本切片 (chunk_size=500, overlap=100)")
    chunks = split_documents(documents, chunk_size=500, chunk_overlap=100)
    print(f"   产生 {len(chunks)} 个切片")

    store = VectorStore()
    if reset:
        print("🗑️  清空向量库")
        store.delete_collection()

    print("📥 写入向量库")
    count = store.add_documents(chunks)
    print(f"   写入 {count} 条记录，总计 {store.count} 条")
    print("✅ 灌入完成\n")


def _print_result(result: "QueryResult") -> None:
    """格式化输出查询结果。"""
    print(f"\n{'='*60}")
    print(f"📍 路由决策: {result.route}")
    print(f"⏱️  耗时: {result.elapsed_seconds}s")
    if result.token_usage:
        print(f"🔢 Token: {result.token_usage}")
    print(f"{'='*60}")
    print(f"\n{result.answer}\n")


def _interactive_mode(crew: RAGCrew) -> None:
    """交互式多轮对话模式。"""
    print("🤖 Agentic RAG 交互模式（输入 quit 或 exit 退出）\n")

    while True:
        try:
            question = input("❓ 请输入问题: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        try:
            result = crew.query(question)
            _print_result(result)
        except Exception as e:
            print(f"\n❌ 查询出错: {e}\n")


def main() -> None:
    """CLI 主入口。"""
    parser = argparse.ArgumentParser(
        description="CrewAI Agentic RAG 系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  uv run python main.py --query '什么是装饰器？'\n"
            "  uv run python main.py --ingest --data-dir data\n"
            "  uv run python main.py  # 交互模式"
        ),
    )
    parser.add_argument("--query", "-q", type=str, help="单次查询问题")
    parser.add_argument("--ingest", action="store_true", help="执行文档灌入")
    parser.add_argument("--data-dir", default="data", help="文档目录（默认 data）")
    parser.add_argument("--reset", action="store_true", help="灌入前清空向量库")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示 Agent 详细思考过程")
    parser.add_argument("--no-memory", action="store_true", help="禁用多轮对话记忆")
    args = parser.parse_args()

    # 文档灌入模式
    if args.ingest:
        _run_ingest(args.data_dir, args.reset)
        return

    # 初始化 Crew
    crew = RAGCrew(verbose=args.verbose, memory=not args.no_memory)

    if args.query:
        # 单次查询模式
        result = crew.query(args.query)
        _print_result(result)
    else:
        # 交互式多轮对话
        _interactive_mode(crew)


if __name__ == "__main__":
    main()
