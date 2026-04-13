"""M2a 测试：向量库存取与检索。"""

from pathlib import Path

import pytest

from src.chunker import split_documents
from src.document_loader import Document, load_directory, load_file
from src.vector_store import VectorStore

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture()
def store(tmp_path):
    """在 tmp_path 中创建隔离的向量库，pytest 自动清理。"""
    return VectorStore(persist_dir=tmp_path / "chroma", collection_name="test")


class TestVectorStore:
    """向量库基本操作测试。"""

    def test_add_and_count(self, store):
        """写入文档后 count 应增加。"""
        chunks = [
            Document(content="Python 是一门编程语言", metadata={"source": "a.txt", "chunk_index": 0}),
            Document(content="Python 支持面向对象", metadata={"source": "a.txt", "chunk_index": 1}),
        ]
        count = store.add_documents(chunks)
        assert count == 2
        assert store.count == 2

    def test_add_empty_returns_zero(self, store):
        """写入空列表应返回 0。"""
        assert store.add_documents([]) == 0

    def test_search_returns_results(self, store):
        """查询应返回语义相关的结果。"""
        chunks = [
            Document(
                content="Python decorator is a powerful feature that allows you to modify function behavior",
                metadata={"source": "a.txt", "chunk_index": 0},
            ),
            Document(
                content="The weather forecast for tomorrow shows heavy rainfall and strong winds",
                metadata={"source": "b.txt", "chunk_index": 0},
            ),
        ]
        store.add_documents(chunks)
        results = store.search("What is a Python decorator?", top_k=2)
        assert len(results) > 0
        assert results[0]["source"] == "a.txt"

    def test_search_top_k(self, store):
        """top_k 应限制返回数量。"""
        chunks = [
            Document(content=f"文档内容 {i}", metadata={"source": f"doc{i}.txt", "chunk_index": 0})
            for i in range(5)
        ]
        store.add_documents(chunks)
        results = store.search("文档内容", top_k=2)
        assert len(results) == 2

    def test_delete_collection_clears_data(self, store):
        """delete_collection 应清空所有数据。"""
        chunks = [Document(content="测试内容", metadata={"source": "a.txt", "chunk_index": 0})]
        store.add_documents(chunks)
        assert store.count == 1
        store.delete_collection()
        assert store.count == 0


class TestIntegration:
    """端到端集成测试：文档加载 → 切片 → 存储 → 检索。"""

    def test_ingest_and_search_decorator(self, store):
        """灌入 Python 文档后查询「装饰器」应命中相关内容。"""
        # 加载
        md_docs = load_file(DATA_DIR / "sample.md")
        pdf_docs = load_file(DATA_DIR / "sample.pdf")
        all_docs = md_docs + pdf_docs

        # 切片
        chunks = split_documents(all_docs, chunk_size=500, chunk_overlap=100)
        assert len(chunks) > 0

        # 存储
        store.add_documents(chunks)
        assert store.count == len(chunks)

        # 检索
        results = store.search("什么是装饰器", top_k=3)
        assert len(results) > 0
        # top-3 中至少有 1 个包含 "decorator" 或 "装饰器"
        hit = any(
            "decorator" in r["content"].lower() or "装饰器" in r["content"]
            for r in results
        )
        assert hit, f"未找到装饰器相关内容，返回: {[r['content'][:50] for r in results]}"

    def test_full_pipeline_from_directory(self, store):
        """完整流水线：加载目录 → 切片 → 存储 → 检索。"""
        docs = load_directory(DATA_DIR)
        chunks = split_documents(docs)
        store.add_documents(chunks)

        assert store.count > 0
        results = store.search("Python 编程语言", top_k=3)
        assert len(results) > 0
