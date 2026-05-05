"""向量检索工具 — 包装 M2a 的向量库检索接口为 CrewAI Tool。

当 M2a 的 vector_store 模块就绪后，自动使用真实检索。
mock 数据仅在显式设置 VECTOR_SEARCH_ENABLE_MOCK_FALLBACK=1 时启用。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from crewai.tools import tool

if TYPE_CHECKING:
    from src.vector_store import VectorStore


def _try_import_vector_store() -> type[VectorStore] | None:
    """尝试导入 M2a 的向量库模块。"""
    try:
        from src.vector_store import VectorStore  # noqa: F401
        return VectorStore
    except (ImportError, ModuleNotFoundError):
        return None


def _mock_search(query: str, top_k: int = 3) -> list[str]:
    """Mock 检索结果，用于 M2a 未完成时的独立测试。"""
    return [
        f"[Mock 结果 1] 与「{query}」相关的文档片段：这是一段示例内容。",
        f"[Mock 结果 2] 与「{query}」相关的文档片段：这是另一段示例内容。",
        f"[Mock 结果 3] 与「{query}」相关的文档片段：这是第三段示例内容。",
    ][:top_k]


def _mock_fallback_enabled() -> bool:
    """是否允许开发/测试环境使用 mock 检索结果。"""
    return os.getenv("VECTOR_SEARCH_ENABLE_MOCK_FALLBACK", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@tool("vector_search_tool")
def vector_search_tool(query: str, top_k: int = 3) -> str:
    """从向量知识库中检索与查询最相关的文档片段。

    当用户的问题需要基于文档内容来回答时，使用此工具检索相关信息。
    返回最相关的文档片段列表，每个片段用换行分隔。

    Args:
        query: 用户的查询问题或关键词。
        top_k: 返回的最相关文档片段数量，默认为 3。
    """
    VectorStore = _try_import_vector_store()

    if VectorStore is None:
        if _mock_fallback_enabled():
            return "\n\n---\n\n".join(_mock_search(query, top_k))
        return (
            "向量检索不可用：未能加载向量库模块。"
            "请确认 src.vector_store 可导入并已完成文档灌入。"
        )

    try:
        store = VectorStore()
        results = store.search(query, top_k=top_k)
        if results:
            return "\n\n---\n\n".join(r["content"] for r in results)
        return "未找到相关文档内容。"
    except Exception as e:
        if _mock_fallback_enabled():
            return f"向量检索出错: {e}，回退到 mock 数据。\n" + "\n\n---\n\n".join(
                _mock_search(query, top_k)
            )
        return f"向量检索不可用：{e}。请检查向量库配置、Embedding 配置或本地索引状态。"
