"""向量库封装模块。

基于 ChromaDB 实现文档的存储与相似度检索。
支持通过 get_embedding_config() 配置外部 Embedding API，
未配置时回退到 ChromaDB 内置的 all-MiniLM-L6-v2 本地 embedding。
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction

from src.document_loader import Document


_DEFAULT_PERSIST_DIR = ".chroma"
_DEFAULT_COLLECTION = "documents"


def _build_embedding_function() -> EmbeddingFunction | None:
    """根据 get_embedding_config() 构建 ChromaDB embedding function。

    Returns:
        配置了外部 API 的 EmbeddingFunction，或 None（使用 ChromaDB 默认）。
    """
    try:
        from src.config import get_embedding_config
    except ImportError:
        return None

    cfg = get_embedding_config()
    api_key = cfg["config"].get("api_key")
    if not api_key:
        return None

    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    return OpenAIEmbeddingFunction(
        model_name=cfg["config"].get("model_name", "text-embedding-v3"),
        api_key=api_key,
        api_base=cfg["config"].get("api_base"),
    )


class VectorStore:
    """ChromaDB 向量库封装。"""

    def __init__(
        self,
        persist_dir: str | Path = _DEFAULT_PERSIST_DIR,
        collection_name: str = _DEFAULT_COLLECTION,
        embedding_fn: EmbeddingFunction | None = None,
    ) -> None:
        """初始化向量库。

        Args:
            persist_dir: 持久化目录路径。
            collection_name: 集合名称。
            embedding_fn: 自定义 embedding function，为 None 时自动从配置构建。
        """
        self._client = chromadb.PersistentClient(path=str(persist_dir))

        # 优先使用传入的 embedding_fn，其次从配置构建，最后用 ChromaDB 默认
        ef = embedding_fn or _build_embedding_function()
        kwargs: dict = {
            "name": collection_name,
            "metadata": {"hnsw:space": "cosine"},
        }
        if ef is not None:
            kwargs["embedding_function"] = ef

        self._collection = self._client.get_or_create_collection(**kwargs)

    @property
    def count(self) -> int:
        """当前集合中的文档数量。"""
        return self._collection.count()

    def add_documents(self, chunks: list[Document]) -> int:
        """批量写入文档切片。

        Args:
            chunks: Document 列表，metadata 中应包含 source 和 chunk_index。

        Returns:
            写入的文档数量。
        """
        if not chunks:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            chunk_index = chunk.metadata.get("chunk_index", 0)
            page = chunk.metadata.get("page")
            doc_id = f"{source}::p{page}::chunk_{chunk_index}" if page else f"{source}::chunk_{chunk_index}"
            ids.append(doc_id)
            documents.append(chunk.content)
            # ChromaDB metadata 只接受 str/int/float/bool
            meta = {}
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            metadatas.append(meta)

        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return len(ids)

    def search(self, query_text: str, top_k: int = 3) -> list[dict]:
        """相似度检索。

        Args:
            query_text: 查询文本。
            top_k: 返回的最相似结果数。

        Returns:
            结果列表，每个包含 content、source、score。
        """
        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(top_k, self.count) if self.count > 0 else 1,
        )

        output: list[dict] = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "content": doc,
                    "source": results["metadatas"][0][i].get("source", "unknown"),
                    "score": results["distances"][0][i] if results["distances"] else None,
                })
        return output

    def delete_collection(self) -> None:
        """清空当前集合的所有数据。"""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )
