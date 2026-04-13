"""Crew 流水线模块 — 组装 Router → Retriever → Responder 三级顺序执行。

支持两种执行路径：
- RETRIEVE 路径：Router 判断需要检索 → Retriever 检索文档 → Responder 生成回答
- DIRECT 路径：Router 判断可直接回答 → Responder 直接生成回答

配置：
- process=Process.sequential（顺序执行）
- memory=True（短期记忆，支持多轮对话上下文）

兼容性说明：
    DeepSeek API 支持 response_format: {"type": "json_object"}，
    但不支持 {"type": "json_schema"}（OpenAI structured output）。
    CrewAI Memory 内部通过 supports_function_calling() 判断是否走
    beta.chat.completions.parse（会发送 json_schema），因此需要为
    Memory 提供一个 wrapper LLM，让它走纯文本 + JSON 解析的降级路径。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from crewai import Crew, Process
from crewai.memory.unified_memory import Memory

from src.agents import (
    create_responder_agent,
    create_retriever_agent,
    create_router_agent,
)
from src.config import get_embedding_config, get_llm
from src.tasks import (
    create_direct_responder_task,
    create_responder_task,
    create_retriever_task,
    create_router_task,
)


@dataclass
class QueryResult:
    """查询结果数据类。

    Attributes:
        answer: 最终回答文本。
        route: 路由决策（RETRIEVE 或 DIRECT）。
        token_usage: token 消耗统计。
        elapsed_seconds: 查询耗时（秒）。
        raw_output: CrewAI 原始输出对象。
    """

    answer: str
    route: str
    token_usage: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    raw_output: Any = None


class _MemoryLLMWrapper:
    """LLM wrapper，让 Memory 走纯文本 + JSON 解析路径。

    CrewAI Memory 的 analyze.py 通过 supports_function_calling() 判断是否
    使用 response_model（structured output / json_schema）。DeepSeek 不支持
    json_schema，但支持 json_object。

    此 wrapper 将 supports_function_calling() 返回 False，迫使 Memory 走
    普通 llm.call() → 返回纯文本 → json.loads() 解析的降级路径。
    同时在 system prompt 中注入 JSON 输出指令，配合 DeepSeek 的 json_object
    模式确保输出合法 JSON。
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def supports_function_calling(self) -> bool:
        """返回 False，绕过 json_schema structured output。"""
        return False

    def call(self, messages: Any, **kwargs: Any) -> Any:
        """转发调用到底层 LLM，忽略 response_model 参数。"""
        # 移除 response_model，走纯文本路径
        kwargs.pop("response_model", None)

        # 在 system prompt 中注入 JSON 输出要求
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict) and first.get("role") == "system":
                first["content"] += "\n\nIMPORTANT: You MUST respond with valid JSON only. No markdown, no extra text."

        return self._llm.call(messages, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """代理其他属性到底层 LLM。"""
        return getattr(self._llm, name)


def _build_embedder_config() -> dict | None:
    """从 embedding 配置构建 CrewAI embedder 字典。

    Returns:
        embedder 配置字典，或 None（使用默认）。
    """
    try:
        cfg = get_embedding_config()
        api_key = cfg["config"].get("api_key")
        if not api_key:
            return None
        return {
            "provider": "openai",
            "config": {
                "model": cfg["config"].get("model_name", "text-embedding-v3"),
                "api_key": api_key,
                "api_base": cfg["config"].get("api_base"),
            },
        }
    except Exception:
        return None


class RAGCrew:
    """Agentic RAG Crew 封装。

    组装三个 Agent（Router / Retriever / Responder），
    根据路由结果选择执行路径，支持多轮对话记忆。
    """

    def __init__(self, verbose: bool = False, memory: bool = True) -> None:
        """初始化 RAG Crew。

        Args:
            verbose: 是否输出详细的 Agent 思考过程。
            memory: 是否启用短期记忆（CrewAI Memory）。
        """
        self._verbose = verbose
        self._memory = memory
        self._llm = get_llm()

        # 创建 Agent 实例（复用同一个 LLM）
        self._router = create_router_agent(llm=self._llm)
        self._retriever = create_retriever_agent(llm=self._llm)
        self._responder = create_responder_agent(llm=self._llm)

        # embedder 配置
        self._embedder = _build_embedder_config()

        # 构建 Memory 实例
        self._memory_instance = self._build_memory() if memory else False

    def _build_memory(self) -> Memory:
        """构建 Memory 实例，兼容 DeepSeek API。

        DeepSeek 不支持 json_schema（structured output），但 CrewAI Memory
        内部会在 supports_function_calling()=True 时使用 response_model。
        通过 _MemoryLLMWrapper 包装，让 Memory 走纯文本 + JSON 解析路径。

        Returns:
            配置好的 Memory 实例。
        """
        memory_llm = _MemoryLLMWrapper(self._llm)
        kwargs: dict[str, Any] = {"llm": memory_llm}
        if self._embedder:
            kwargs["embedder"] = self._embedder
        return Memory(**kwargs)

    def query(self, question: str) -> QueryResult:
        """执行完整的 RAG 查询流程。

        流程：
        1. Router 判断问题类型（RETRIEVE / DIRECT）
        2. 根据路由结果选择执行路径
        3. 返回最终回答及性能指标

        Args:
            question: 用户的问题。

        Returns:
            QueryResult 包含回答、路由决策、token 消耗和耗时。
        """
        start = time.time()

        # Step 1: 路由判断
        route = self._run_router(question)

        # Step 2: 根据路由选择执行路径
        if route == "RETRIEVE":
            result = self._run_retrieve_pipeline(question)
        else:
            result = self._run_direct_pipeline(question)

        elapsed = time.time() - start

        # 提取 token 使用信息
        token_usage = {}
        if result and hasattr(result, "token_usage"):
            token_usage = dict(result.token_usage) if result.token_usage else {}

        answer = str(result) if result else "无法生成回答。"

        return QueryResult(
            answer=answer,
            route=route,
            token_usage=token_usage,
            elapsed_seconds=round(elapsed, 2),
            raw_output=result,
        )

    def _run_router(self, question: str) -> str:
        """运行路由 Agent，判断问题类型。

        Args:
            question: 用户问题。

        Returns:
            "RETRIEVE" 或 "DIRECT"。
        """
        router_task = create_router_task(self._router, question)

        crew = Crew(
            agents=[self._router],
            tasks=[router_task],
            process=Process.sequential,
            verbose=self._verbose,
        )
        result = crew.kickoff()

        # 解析路由结果
        raw = str(result).strip().upper()
        if "RETRIEVE" in raw:
            return "RETRIEVE"
        return "DIRECT"

    def _run_retrieve_pipeline(self, question: str) -> Any:
        """运行检索路径：Router → Retriever → Responder。

        Args:
            question: 用户问题。

        Returns:
            CrewOutput 结果。
        """
        router_task = create_router_task(self._router, question)
        retriever_task = create_retriever_task(
            self._retriever, question, router_task
        )
        responder_task = create_responder_task(
            self._responder, question, retriever_task
        )

        crew = Crew(
            agents=[self._router, self._retriever, self._responder],
            tasks=[router_task, retriever_task, responder_task],
            process=Process.sequential,
            verbose=self._verbose,
            memory=self._memory_instance,
        )
        return crew.kickoff()

    def _run_direct_pipeline(self, question: str) -> Any:
        """运行直答路径：Responder 直接回答。

        Args:
            question: 用户问题。

        Returns:
            CrewOutput 结果。
        """
        direct_task = create_direct_responder_task(self._responder, question)

        crew = Crew(
            agents=[self._responder],
            tasks=[direct_task],
            process=Process.sequential,
            verbose=self._verbose,
            memory=self._memory_instance,
        )
        return crew.kickoff()
