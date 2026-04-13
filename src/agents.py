"""Agent 定义模块 — Router / Retriever / Responder 三级架构。

Router:    判断问题是否需要检索，还是直接回答。
Retriever: 从向量库检索相关文档片段。
Responder: 基于检索结果生成带引用的回答。
"""

from __future__ import annotations

from crewai import Agent

from src.config import get_llm
from src.tools.vector_search_tool import vector_search_tool


def create_router_agent(llm=None) -> Agent:
    """创建意图路由 Agent。

    纯推理角色，不配备工具。根据问题内容判断：
    - 需要检索文档 → 输出 "RETRIEVE"
    - 可以直接回答 → 输出 "DIRECT"

    Args:
        llm: 可选的 LLM 实例，默认使用 config 中的配置。
    """
    return Agent(
        role="意图路由器",
        goal="准确判断用户问题是否需要从知识库检索文档来回答",
        backstory=(
            "你是一个经验丰富的问题分类专家。"
            "你能快速判断一个问题是通用常识问题（可以直接回答），"
            "还是需要查阅特定文档/知识库才能准确回答的专业问题。"
            "你的判断直接影响后续流程的效率。"
        ),
        llm=llm or get_llm(),
        verbose=False,
        allow_delegation=False,
    )


def create_retriever_agent(llm=None) -> Agent:
    """创建知识检索 Agent。

    配备 vector_search_tool，负责从向量库中检索相关文档片段。

    Args:
        llm: 可选的 LLM 实例，默认使用 config 中的配置。
    """
    return Agent(
        role="知识检索员",
        goal="从知识库中检索出与用户问题最相关的文档片段",
        backstory=(
            "你是一个专业的信息检索专家。"
            "你擅长将用户的问题转化为有效的检索查询，"
            "并从知识库中找到最相关、最有价值的文档片段。"
            "你会尝试多种查询策略以确保检索结果的全面性。"
        ),
        llm=llm or get_llm(),
        tools=[vector_search_tool],
        verbose=False,
        allow_delegation=False,
    )


def create_responder_agent(llm=None) -> Agent:
    """创建回答生成 Agent。

    不配备工具，基于检索结果（或直接知识）生成最终回答。

    Args:
        llm: 可选的 LLM 实例，默认使用 config 中的配置。
    """
    return Agent(
        role="回答生成器",
        goal="基于提供的信息生成准确、完整、带引用的中文回答",
        backstory=(
            "你是一个专业的知识问答助手。"
            "你擅长将检索到的文档片段整合为连贯、准确的回答。"
            "你总是会标注信息来源，确保回答可追溯。"
            "当信息不足时，你会诚实说明而不是编造内容。"
        ),
        llm=llm or get_llm(),
        verbose=False,
        allow_delegation=False,
    )
