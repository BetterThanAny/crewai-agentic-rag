"""M2b 测试：Agent 与 Task 单元测试。

通过 mock create_llm 跳过真实 LLM 初始化，验证：
1. Agent 创建正确（角色、工具配置）
2. Task 创建正确（描述、期望输出）
3. Router 路由逻辑（RETRIEVE / DIRECT）
"""

from unittest.mock import MagicMock, patch

from crewai import Agent, Task

from src.agents import (
    create_responder_agent,
    create_retriever_agent,
    create_router_agent,
)
from src.tasks import (
    create_direct_responder_task,
    create_responder_task,
    create_retriever_task,
    create_router_task,
)


def _make_mock_llm() -> MagicMock:
    """创建 mock LLM，绕过 CrewAI 的模型验证。"""
    mock = MagicMock()
    mock.model = "mock-model"
    mock.supports_function_calling.return_value = True
    mock.call.return_value = "mock response"
    return mock


def _create_agent_with_mock_llm(create_fn, **kwargs):
    """使用 mock LLM 创建 Agent，绕过 create_llm 验证。"""
    mock_llm = _make_mock_llm()
    with patch("crewai.agent.core.create_llm", return_value=mock_llm):
        return create_fn(llm="mock-model", **kwargs)


class TestAgentCreation:
    """Agent 创建测试。"""

    def test_router_agent_has_no_tools(self):
        """Router Agent 不应配备任何工具。"""
        agent = _create_agent_with_mock_llm(create_router_agent)
        assert isinstance(agent, Agent)
        assert agent.role == "意图路由器"
        assert agent.tools == []

    def test_retriever_agent_has_vector_search_tool(self):
        """Retriever Agent 应配备 vector_search_tool。"""
        agent = _create_agent_with_mock_llm(create_retriever_agent)
        assert isinstance(agent, Agent)
        assert agent.role == "知识检索员"
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "vector_search_tool"

    def test_responder_agent_has_no_tools(self):
        """Responder Agent 不应配备工具。"""
        agent = _create_agent_with_mock_llm(create_responder_agent)
        assert isinstance(agent, Agent)
        assert agent.role == "回答生成器"
        assert agent.tools == []

    def test_all_agents_disable_delegation(self):
        """所有 Agent 应禁用委派。"""
        for create_fn in [create_router_agent, create_retriever_agent, create_responder_agent]:
            agent = _create_agent_with_mock_llm(create_fn)
            assert agent.allow_delegation is False


class TestTaskCreation:
    """Task 创建测试。"""

    def test_router_task_contains_question(self):
        """Router Task 描述中应包含用户问题。"""
        agent = _create_agent_with_mock_llm(create_router_agent)
        question = "什么是机器学习？"
        task = create_router_task(agent, question)
        assert isinstance(task, Task)
        assert question in task.description
        assert "RETRIEVE" in task.description
        assert "DIRECT" in task.description

    def test_retriever_task_has_context(self):
        """Retriever Task 应以 Router Task 为上下文。"""
        router = _create_agent_with_mock_llm(create_router_agent)
        retriever = _create_agent_with_mock_llm(create_retriever_agent)
        question = "文档中提到了哪些算法？"

        router_task = create_router_task(router, question)
        retriever_task = create_retriever_task(retriever, question, router_task)

        assert retriever_task.context == [router_task]
        assert "vector_search_tool" in retriever_task.description

    def test_responder_task_has_context(self):
        """Responder Task 应以 Retriever Task 为上下文。"""
        router = _create_agent_with_mock_llm(create_router_agent)
        retriever = _create_agent_with_mock_llm(create_retriever_agent)
        responder = _create_agent_with_mock_llm(create_responder_agent)
        question = "解释一下向量数据库"

        router_task = create_router_task(router, question)
        retriever_task = create_retriever_task(retriever, question, router_task)
        responder_task = create_responder_task(responder, question, retriever_task)

        assert responder_task.context == [retriever_task]
        assert "中文" in responder_task.description

    def test_direct_responder_task_has_no_context(self):
        """Direct Responder Task 不应有上下文依赖。"""
        responder = _create_agent_with_mock_llm(create_responder_agent)
        task = create_direct_responder_task(responder, "1+1等于几？")
        # CrewAI 使用 NOT_SPECIFIED 作为未设置 context 的默认值
        from crewai.task import NOT_SPECIFIED
        assert task.context is NOT_SPECIFIED or task.context is None or task.context == []

    def test_all_tasks_have_expected_output(self):
        """所有 Task 应定义 expected_output。"""
        router = _create_agent_with_mock_llm(create_router_agent)
        retriever = _create_agent_with_mock_llm(create_retriever_agent)
        responder = _create_agent_with_mock_llm(create_responder_agent)
        question = "测试问题"

        router_task = create_router_task(router, question)
        retriever_task = create_retriever_task(retriever, question, router_task)
        responder_task = create_responder_task(responder, question, retriever_task)
        direct_task = create_direct_responder_task(responder, question)

        for task in [router_task, retriever_task, responder_task, direct_task]:
            assert task.expected_output
            assert len(task.expected_output) > 0


class TestRouterLogic:
    """Router 路由逻辑测试。"""

    def test_router_task_expects_retrieve_or_direct(self):
        """Router Task 的 expected_output 应包含 RETRIEVE 和 DIRECT。"""
        agent = _create_agent_with_mock_llm(create_router_agent)
        task = create_router_task(agent, "随便问个问题")
        assert "RETRIEVE" in task.expected_output
        assert "DIRECT" in task.expected_output

    def test_retrieve_question_description_mentions_criteria(self):
        """Router Task 描述应包含判断标准。"""
        agent = _create_agent_with_mock_llm(create_router_agent)
        task = create_router_task(agent, "文档里怎么说的？")
        assert "判断标准" in task.description
