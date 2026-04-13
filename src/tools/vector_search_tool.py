"""向量检索工具 — 包装 M2a 的向量库检索接口为 CrewAI Tool。

当 M2a 的 vector_store 模块就绪后，自动使用真实检索；
否则回退到 mock 数据，保证 M2b 可独立开发和测试。
"""

from __future__ import annotations

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

    if VectorStore is not None:
        try:
            store = VectorStore()
            results = store.search(query, top_k=top_k)
            if results:
                return "\n\n---\n\n".join(r["content"] for r in results)
            return "未找到相关文档内容。"
        except Exception as e:
            return f"向量检索出错: {e}，回退到 mock 数据。\n" + "\n\n---\n\n".join(
                _mock_search(query, top_k)
            )
    else:
        # M2a 未就绪，使用 mock
        return "\n\n---\n\n".join(_mock_search(query, top_k))
