"""Task 定义模块 — 为 Router / Retriever / Responder 定义对应任务。

每个 Task 包含：description（描述）、expected_output（期望输出格式）、agent（负责执行的 Agent）。
Task 之间通过 context 参数传递上游输出。
"""

from __future__ import annotations

from crewai import Agent, Task


def create_router_task(agent: Agent, question: str) -> Task:
    """创建路由判断任务。

    Args:
        agent: 路由 Agent 实例。
        question: 用户输入的问题。
    """
    return Task(
        description=(
            f"分析以下用户问题，判断是否需要从知识库检索文档来回答。\n\n"
            f"用户问题：{question}\n\n"
            f"判断标准：\n"
            f"- 如果问题涉及特定文档、专业知识或需要引用具体内容 → 输出 RETRIEVE\n"
            f"- 如果问题是通用常识、简单计算或闲聊 → 输出 DIRECT\n\n"
            f"只输出 RETRIEVE 或 DIRECT 其中一个词，不要输出其他内容。"
        ),
        expected_output="RETRIEVE 或 DIRECT（单个词）",
        agent=agent,
    )


def create_retriever_task(agent: Agent, question: str, router_task: Task) -> Task:
    """创建知识检索任务。

    Args:
        agent: 检索 Agent 实例。
        question: 用户输入的问题。
        router_task: 路由任务实例，作为上下文输入。
    """
    return Task(
        description=(
            f"根据用户问题从知识库中检索相关文档片段。\n\n"
            f"用户问题：{question}\n\n"
            f"要求：\n"
            f"1. 使用 vector_search_tool 检索相关文档\n"
            f"2. 可以尝试不同的查询关键词以提高召回率\n"
            f"3. 将检索到的所有相关片段整理输出"
        ),
        expected_output="与问题相关的文档片段列表，每个片段独立呈现",
        agent=agent,
        context=[router_task],
    )


def create_responder_task(
    agent: Agent, question: str, retriever_task: Task
) -> Task:
    """创建回答生成任务。

    Args:
        agent: 回答生成 Agent 实例。
        question: 用户输入的问题。
        retriever_task: 检索任务实例，作为上下文输入。
    """
    return Task(
        description=(
            f"基于检索到的文档片段，回答用户的问题。\n\n"
            f"用户问题：{question}\n\n"
            f"要求：\n"
            f"1. 使用中文回答\n"
            f"2. 回答要准确、完整、有条理\n"
            f"3. 引用文档内容时标注来源\n"
            f"4. 如果检索结果不足以回答问题，诚实说明"
        ),
        expected_output="结构清晰、带引用来源的中文回答",
        agent=agent,
        context=[retriever_task],
    )


def create_direct_responder_task(agent: Agent, question: str) -> Task:
    """创建直接回答任务（Router 判断为 DIRECT 时使用）。

    Args:
        agent: 回答生成 Agent 实例。
        question: 用户输入的问题。
    """
    return Task(
        description=(
            f"直接回答用户的问题（无需检索文档）。\n\n"
            f"用户问题：{question}\n\n"
            f"要求：\n"
            f"1. 使用中文回答\n"
            f"2. 回答要准确、简洁\n"
            f"3. 如果不确定答案，诚实说明"
        ),
        expected_output="简洁准确的中文回答",
        agent=agent,
    )
