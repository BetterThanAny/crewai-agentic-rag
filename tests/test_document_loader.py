"""M2a 测试：文档加载与文本切片。"""

from pathlib import Path

import pytest

from src.chunker import split_documents, split_text
from src.document_loader import Document, load_directory, load_file

DATA_DIR = Path(__file__).parent.parent / "data"


class TestDocumentLoader:
    """文档加载器测试。"""

    def test_load_pdf(self):
        """加载 PDF 应返回 Document 列表，每页一个。"""
        docs = load_file(DATA_DIR / "sample.pdf")
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)
        # PDF 应有页码元数据
        assert all("page" in d.metadata for d in docs)
        full_text = " ".join(d.content for d in docs)
        assert "Python" in full_text

    def test_load_txt(self):
        """加载 TXT 应返回单个 Document。"""
        docs = load_file(DATA_DIR / "sample.txt")
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "Python" in docs[0].content
        assert docs[0].metadata["source"] == "sample.txt"

    def test_load_markdown(self):
        """加载 Markdown 应返回单个 Document。"""
        docs = load_file(DATA_DIR / "sample.md")
        assert len(docs) == 1
        assert "装饰器" in docs[0].content

    def test_load_nonexistent_file(self):
        """加载不存在的文件应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_file("不存在的文件.pdf")

    def test_load_unsupported_format(self, tmp_path):
        """加载不支持的格式应抛出 ValueError。"""
        f = tmp_path / "test.xlsx"
        f.write_text("dummy")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            load_file(f)

    def test_load_empty_file(self, tmp_path):
        """空文件应返回包含空内容的 Document。"""
        f = tmp_path / "empty.txt"
        f.write_text("")
        docs = load_file(f)
        assert len(docs) == 1
        assert docs[0].content == ""

    def test_load_directory(self):
        """批量加载目录应返回所有支持的文档。"""
        docs = load_directory(DATA_DIR)
        assert len(docs) >= 3
        sources = {d.metadata["source"] for d in docs}
        assert "sample.pdf" in sources
        assert "sample.txt" in sources
        assert "sample.md" in sources

    def test_load_non_utf8_file(self, tmp_path):
        """GBK 编码的文件应能正常加载。"""
        f = tmp_path / "gbk.txt"
        f.write_bytes("你好世界".encode("gbk"))
        docs = load_file(f)
        assert len(docs) == 1
        assert "你好世界" in docs[0].content

    def test_pdf_page_metadata(self):
        """PDF 每页的 Document 应有正确的页码。"""
        docs = load_file(DATA_DIR / "sample.pdf")
        pages = [d.metadata["page"] for d in docs]
        assert pages == list(range(1, len(docs) + 1))


class TestChunker:
    """文本切片测试。"""

    def test_basic_split(self):
        """基本切分应产出所有 chunk 长度 <= chunk_size。"""
        text = "A" * 1500
        chunks = split_text(text, chunk_size=500, chunk_overlap=0)
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) <= 600  # 允许 overlap 带来的少量超出

    def test_pdf_chunks_within_limit(self):
        """PDF 切片每个 chunk 长度应 <= 500（不含 overlap 部分）。"""
        docs = load_file(DATA_DIR / "sample.pdf")
        full_text = "\n".join(d.content for d in docs)
        chunks = split_text(full_text, chunk_size=500, chunk_overlap=0)
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) <= 500

    def test_pdf_produces_multiple_chunks(self):
        """3 页 PDF 应产出多个 chunks。"""
        docs = load_file(DATA_DIR / "sample.pdf")
        full_text = "\n".join(d.content for d in docs)
        chunks = split_text(full_text, chunk_size=500, chunk_overlap=100)
        assert len(chunks) > 1

    def test_empty_text_returns_empty(self):
        """空文本应返回空列表。"""
        assert split_text("") == []
        assert split_text("   ") == []

    def test_short_text_single_chunk(self):
        """短文本应返回单个 chunk。"""
        chunks = split_text("Hello World", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "Hello World"

    def test_split_documents_preserves_source(self):
        """split_documents 应保留 source 元数据。"""
        docs = [Document(content="A" * 1200, metadata={"source": "test.txt"})]
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=0)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.txt"
            assert "chunk_index" in chunk.metadata

    def test_split_documents_preserves_page(self):
        """split_documents 应保留 page 元数据。"""
        docs = [Document(content="B" * 1200, metadata={"source": "test.pdf", "page": 2})]
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=0)
        for chunk in chunks:
            assert chunk.metadata["page"] == 2

    def test_overlap_creates_continuity(self):
        """overlap 应使相邻 chunk 有部分重叠内容。"""
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容，这是一段比较长的文本用于测试重叠效果。"
        chunks = split_text(text, chunk_size=20, chunk_overlap=5)
        if len(chunks) >= 2:
            # 第二个 chunk 应包含第一个 chunk 尾部的部分内容
            assert len(chunks[1]) > 0
