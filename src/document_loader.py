"""多格式文档加载模块。

支持 PDF、TXT、Markdown 格式，统一返回 Document 数据类列表。
"""

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf


@dataclass
class Document:
    """文档数据类，承载文本内容与元数据。

    Attributes:
        content: 文档的纯文本内容。
        metadata: 元数据字典，至少包含 source 字段。
    """

    content: str
    metadata: dict[str, object] = field(default_factory=dict)


def load_file(file_path: str | Path) -> list[Document]:
    """加载文档并返回 Document 列表。

    PDF 按页拆分，每页一个 Document；
    TXT / Markdown 作为整体返回单个 Document。

    Args:
        file_path: 文档路径，支持 .pdf / .txt / .md 格式。

    Returns:
        Document 列表，每个 Document 包含 content 和 metadata。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 不支持的文件格式。
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = path.suffix.lower()
    loaders: dict[str, object] = {
        ".pdf": _load_pdf,
        ".txt": _load_text,
        ".md": _load_text,
    }

    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(f"不支持的文件格式: {suffix}（支持 .pdf / .txt / .md）")

    return loader(path)


def load_directory(dir_path: str | Path) -> list[Document]:
    """批量加载目录下所有支持的文档。

    Args:
        dir_path: 目录路径。

    Returns:
        Document 列表。
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"不是有效目录: {dir_path}")

    supported = {".pdf", ".txt", ".md"}
    results: list[Document] = []
    for f in sorted(dir_path.iterdir()):
        if f.suffix.lower() in supported:
            docs = load_file(f)
            results.extend(doc for doc in docs if doc.content.strip())
    return results


def _load_pdf(path: Path) -> list[Document]:
    """使用 PyMuPDF 提取 PDF，按页拆分为 Document 列表。"""
    documents: list[Document] = []
    with pymupdf.open(str(path)) as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                documents.append(Document(
                    content=text,
                    metadata={"source": path.name, "page": page_num + 1},
                ))
    return documents


def _load_text(path: Path) -> list[Document]:
    """加载纯文本文件，自动检测编码。"""
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            content = path.read_text(encoding=encoding)
            return [Document(
                content=content,
                metadata={"source": path.name},
            )]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "auto", b"", 0, 1, f"无法解码文件: {path}（尝试了 utf-8, gbk, latin-1）"
    )
