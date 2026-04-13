"""文本切片模块。

实现递归字符切片（Recursive Character Text Splitter），
按层级分隔符逐级拆分文本，确保每个 chunk 在 chunk_size 以内且有 overlap 重叠。
"""

from __future__ import annotations

from src.document_loader import Document


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    separators: list[str] | None = None,
) -> list[str]:
    """将文本递归切分为带重叠的片段。

    Args:
        text: 待切分的文本。
        chunk_size: 每个 chunk 的最大字符数。
        chunk_overlap: 相邻 chunk 之间的重叠字符数。
        separators: 分隔符列表，按优先级从高到低排列。

    Returns:
        切分后的文本片段列表。
    """
    if not text or not text.strip():
        return []

    if separators is None:
        separators = ["\n\n", "\n", "。", ".", " ", ""]

    return _recursive_split(text, chunk_size, chunk_overlap, separators)


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """批量切分文档列表。

    Args:
        documents: Document 列表。
        chunk_size: 每个 chunk 的最大字符数。
        chunk_overlap: 相邻 chunk 之间的重叠字符数。

    Returns:
        切分后的 Document 列表，metadata 中新增 chunk_index 字段。
    """
    results: list[Document] = []
    for doc in documents:
        chunks = split_text(doc.content, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            metadata = {**doc.metadata, "chunk_index": i}
            results.append(Document(content=chunk, metadata=metadata))
    return results


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    """递归切分核心逻辑。"""
    # 文本已经足够短
    if len(text) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    # 找到当前层级能用的分隔符
    separator = separators[-1]  # 默认用最后一个（空字符串 = 逐字符）
    for sep in separators:
        if sep and sep in text:
            separator = sep
            break

    # 按分隔符切分
    if separator:
        parts = text.split(separator)
    else:
        # 空字符串分隔符 = 逐字符切割
        parts = list(text)

    # 合并片段，使每个 chunk 不超过 chunk_size
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = current + separator + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # 如果单个 part 超过 chunk_size，用更细粒度分隔符递归处理
            if len(part) > chunk_size and separators.index(separator) < len(separators) - 1:
                sub_chunks = _recursive_split(
                    part, chunk_size, chunk_overlap, separators[separators.index(separator) + 1:]
                )
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = part

    if current and current.strip():
        chunks.append(current.strip())

    # 添加重叠
    if chunk_overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, chunk_overlap)

    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """为相邻 chunk 添加首尾重叠。"""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        merged = prev_tail + " " + chunks[i]
        result.append(merged.strip())
    return result
