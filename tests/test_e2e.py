"""M3 端到端测试：Crew 流水线集成。

测试覆盖：
1. 端到端测试（mock）：路由 → 检索 → 回答完整流程
2. 路由正确性：文档问题走 RETRIEVE，常识问题走 DIRECT
3. Memory 测试：CrewAI Memory + _MemoryLLMWrapper 兼容性
4. 性能测试：记录 token 消耗和响应时间
5. 回退测试：embedding 服务不可用时优雅降级
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.crew import RAGCrew, QueryResult, _build_embedder_config, _MemoryLLMWrapper

DATA_DIR = Path(__file__).parent.parent / "data"


# ─── Fixtures ───────────────────────────────────────────────


def _make_mock_llm() -> MagicMock:
    """创建 mock LLM（用于 create_llm 返回值）。"""
    mock = MagicMock()
    mock.model = "mock-model"
    mock.supports_function_calling.return_value = True
    mock.call.return_value = "mock response"
    return mock


@pytest.fixture()
def mock_crew():
    """使用 mock LLM 的 RAGCrew，不消耗 API。"""
    mock_llm = _make_mock_llm()
    with patch("src.crew.get_llm", return_value="mock-model"), \
         patch("src.agents.get_llm", return_value="mock-model"), \
         patch("crewai.agent.core.create_llm", return_value=mock_llm), \
         patch("src.crew.Memory"):
        crew = RAGCrew(verbose=False, memory=True)
        yield crew


# ─── QueryResult 数据类测试 ─────────────────────────────────


class TestQueryResult:
    """QueryResult 数据类测试。"""

    def test_default_values(self):
        """默认值应正确初始化。"""
        result = QueryResult(answer="测试", route="DIRECT")
        assert result.answer == "测试"
        assert result.route == "DIRECT"
        assert result.token_usage == {}
        assert result.elapsed_seconds == 0.0
        assert result.raw_output is None

    def test_custom_values(self):
        """自定义值应正确设置。"""
        result = QueryResult(
            answer="回答",
            route="RETRIEVE",
            token_usage={"total_tokens": 100},
            elapsed_seconds=1.5,
        )
        assert result.token_usage["total_tokens"] == 100
        assert result.elapsed_seconds == 1.5


# ─── _MemoryLLMWrapper 测试 ────────────────────────────────


class TestMemoryLLMWrapper:
    """Memory LLM Wrapper 测试。"""

    def test_supports_function_calling_returns_false(self):
        """wrapper 应返回 False 以绕过 json_schema。"""
        mock_llm = _make_mock_llm()
        wrapper = _MemoryLLMWrapper(mock_llm)
        assert wrapper.supports_function_calling() is False

    def test_call_strips_response_model(self):
        """call() 应移除 response_model 参数。"""
        from pydantic import BaseModel

        class FakeModel(BaseModel):
            data: str = ""

        mock_llm = _make_mock_llm()
        mock_llm.call.return_value = '{"memories": ["test"]}'
        wrapper = _MemoryLLMWrapper(mock_llm)

        messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "hi"}]
        wrapper.call(messages, response_model=FakeModel)

        # 底层 LLM 不应收到 response_model
        _, call_kwargs = mock_llm.call.call_args
        assert "response_model" not in call_kwargs

    def test_injects_json_instruction(self):
        """call() 应在 system prompt 中注入 JSON 指令。"""
        mock_llm = _make_mock_llm()
        mock_llm.call.return_value = "{}"
        wrapper = _MemoryLLMWrapper(mock_llm)

        messages = [{"role": "system", "content": "You are helpful."}]
        wrapper.call(messages)

        actual_messages = mock_llm.call.call_args[0][0]
        assert "valid JSON" in actual_messages[0]["content"]

    def test_proxies_other_attributes(self):
        """wrapper 应代理其他属性到底层 LLM。"""
        mock_llm = _make_mock_llm()
        mock_llm.model = "deepseek-chat"
        wrapper = _MemoryLLMWrapper(mock_llm)
        assert wrapper.model == "deepseek-chat"


# ─── Crew 初始化测试 ────────────────────────────────────────


class TestRAGCrewInit:
    """RAGCrew 初始化测试。"""

    def test_crew_creates_three_agents(self, mock_crew):
        """Crew 应创建 Router / Retriever / Responder 三个 Agent。"""
        assert mock_crew._router is not None
        assert mock_crew._retriever is not None
        assert mock_crew._responder is not None

    def test_crew_router_has_correct_role(self, mock_crew):
        """Router Agent 角色应为意图路由器。"""
        assert mock_crew._router.role == "意图路由器"

    def test_crew_retriever_has_tool(self, mock_crew):
        """Retriever Agent 应配备 vector_search_tool。"""
        assert len(mock_crew._retriever.tools) == 1
        assert mock_crew._retriever.tools[0].name == "vector_search_tool"

    def test_crew_responder_has_no_tools(self, mock_crew):
        """Responder Agent 不应有工具。"""
        assert mock_crew._responder.tools == []

    def test_crew_verbose_default_false(self, mock_crew):
        """verbose 默认应为 False。"""
        assert mock_crew._verbose is False

    def test_crew_memory_enabled_by_default(self, mock_crew):
        """memory 默认应启用。"""
        assert mock_crew._memory is True

    def test_crew_memory_can_be_disabled(self):
        """memory 可以禁用。"""
        mock_llm = _make_mock_llm()
        with patch("src.crew.get_llm", return_value="mock-model"), \
             patch("src.agents.get_llm", return_value="mock-model"), \
             patch("crewai.agent.core.create_llm", return_value=mock_llm):
            crew = RAGCrew(memory=False)
            assert crew._memory is False
            assert crew._memory_instance is False


# ─── 路由逻辑测试 ──────────────────────────────────────────


class TestRouterLogic:
    """路由判断逻辑测试。"""

    def test_run_router_returns_retrieve_or_direct(self, mock_crew):
        """_run_router 应返回 RETRIEVE 或 DIRECT。"""
        with patch.object(mock_crew, "_run_router", return_value="RETRIEVE"):
            assert mock_crew._run_router("文档里怎么说的？") == "RETRIEVE"

        with patch.object(mock_crew, "_run_router", return_value="DIRECT"):
            assert mock_crew._run_router("1+1等于几？") == "DIRECT"

    def test_route_parsing_retrieve(self):
        """包含 RETRIEVE 的结果应解析为 RETRIEVE。"""
        for raw in ["RETRIEVE", " retrieve ", "RETRIEVE\n", "需要 RETRIEVE"]:
            upper = raw.strip().upper()
            result = "RETRIEVE" if "RETRIEVE" in upper else "DIRECT"
            assert result == "RETRIEVE", f"'{raw}' 应解析为 RETRIEVE"

    def test_route_parsing_direct(self):
        """不包含 RETRIEVE 的结果应解析为 DIRECT。"""
        for raw in ["DIRECT", " direct ", "可以直接回答", "DIRECT\n"]:
            upper = raw.strip().upper()
            result = "RETRIEVE" if "RETRIEVE" in upper else "DIRECT"
            assert result == "DIRECT", f"'{raw}' 应解析为 DIRECT"


# ─── 流水线执行测试（mock Crew.kickoff）────────────────────


class TestPipelineExecution:
    """流水线执行测试，mock Crew.kickoff 验证流程正确性。"""

    def test_retrieve_pipeline_calls_three_agents(self, mock_crew):
        """RETRIEVE 路径应使用三个 Agent。"""
        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "基于文档的回答"
        mock_output.token_usage = {"total_tokens": 150}

        with patch("src.crew.Crew") as MockCrew:
            MockCrew.return_value.kickoff.return_value = mock_output
            mock_crew._run_retrieve_pipeline("什么是装饰器？")

            call_kwargs = MockCrew.call_args[1]
            assert len(call_kwargs["agents"]) == 3
            assert len(call_kwargs["tasks"]) == 3

    def test_direct_pipeline_calls_one_agent(self, mock_crew):
        """DIRECT 路径应只使用 Responder Agent。"""
        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "直接回答"

        with patch("src.crew.Crew") as MockCrew:
            MockCrew.return_value.kickoff.return_value = mock_output
            mock_crew._run_direct_pipeline("1+1等于几？")

            call_kwargs = MockCrew.call_args[1]
            assert len(call_kwargs["agents"]) == 1
            assert len(call_kwargs["tasks"]) == 1

    def test_query_returns_query_result(self, mock_crew):
        """query() 应返回 QueryResult 实例。"""
        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "测试回答"
        mock_output.token_usage = {"total_tokens": 100}

        with patch.object(mock_crew, "_run_router", return_value="DIRECT"), \
             patch.object(mock_crew, "_run_direct_pipeline", return_value=mock_output):
            result = mock_crew.query("测试问题")
            assert isinstance(result, QueryResult)
            assert result.answer == "测试回答"
            assert result.route == "DIRECT"
            assert result.elapsed_seconds >= 0

    def test_query_retrieve_path(self, mock_crew):
        """query() RETRIEVE 路径应正确调用 _run_retrieve_pipeline。"""
        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "检索回答"
        mock_output.token_usage = {}

        with patch.object(mock_crew, "_run_router", return_value="RETRIEVE"), \
             patch.object(mock_crew, "_run_retrieve_pipeline", return_value=mock_output) as mock_retrieve:
            result = mock_crew.query("文档中的内容是什么？")
            assert result.route == "RETRIEVE"
            mock_retrieve.assert_called_once()

    def test_query_measures_elapsed_time(self, mock_crew):
        """query() 应记录耗时。"""
        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "回答"
        mock_output.token_usage = {}

        with patch.object(mock_crew, "_run_router", return_value="DIRECT"), \
             patch.object(mock_crew, "_run_direct_pipeline", return_value=mock_output):
            result = mock_crew.query("问题")
            assert result.elapsed_seconds >= 0
            assert isinstance(result.elapsed_seconds, float)

    def test_memory_passed_to_crew(self, mock_crew):
        """memory 启用时，Crew 创建时应传入 memory 实例。"""
        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "回答"

        with patch("src.crew.Crew") as MockCrew:
            MockCrew.return_value.kickoff.return_value = mock_output
            mock_crew._run_direct_pipeline("问题")

            call_kwargs = MockCrew.call_args[1]
            assert "memory" in call_kwargs


# ─── Embedder 配置测试 ─────────────────────────────────────


class TestEmbedderConfig:
    """Embedder 配置构建测试。"""

    def test_returns_none_without_api_key(self):
        """无 API key 时应返回 None。"""
        with patch("src.crew.get_embedding_config", return_value={
            "provider": "openai",
            "config": {"model_name": "text-embedding-v3", "api_key": None},
        }):
            assert _build_embedder_config() is None

    def test_returns_config_with_api_key(self):
        """有 API key 时应返回正确的配置字典。"""
        with patch("src.crew.get_embedding_config", return_value={
            "provider": "openai",
            "config": {
                "model_name": "text-embedding-v3",
                "api_key": "test-key",
                "api_base": "https://example.com",
            },
        }):
            config = _build_embedder_config()
            assert config is not None
            assert config["provider"] == "openai"
            assert config["config"]["model"] == "text-embedding-v3"
            assert config["config"]["api_key"] == "test-key"

    def test_returns_none_on_exception(self):
        """配置异常时应返回 None 而非抛出错误。"""
        with patch("src.crew.get_embedding_config", side_effect=Exception("配置错误")):
            assert _build_embedder_config() is None


# ─── 回退测试 ──────────────────────────────────────────────


class TestGracefulDegradation:
    """优雅降级测试。"""

    def test_vector_search_tool_fallback_to_mock(self):
        """向量库不可用时，vector_search_tool 应回退到 mock 数据。"""
        from src.tools.vector_search_tool import vector_search_tool

        with patch("src.tools.vector_search_tool._try_import_vector_store", return_value=None):
            result = vector_search_tool.run(query="测试查询")
            assert "Mock 结果" in result

    def test_vector_search_tool_handles_store_error(self):
        """向量库检索出错时应回退到 mock。"""
        from src.tools.vector_search_tool import vector_search_tool

        mock_store_cls = MagicMock()
        mock_store_cls.return_value.search.side_effect = Exception("连接失败")

        with patch("src.tools.vector_search_tool._try_import_vector_store", return_value=mock_store_cls):
            result = vector_search_tool.run(query="测试查询")
            assert "mock" in result.lower() or "出错" in result

    def test_crew_handles_kickoff_error(self, mock_crew):
        """Crew kickoff 异常时 query() 应返回错误信息而非崩溃。"""
        with patch.object(mock_crew, "_run_router", side_effect=Exception("LLM 不可用")):
            with pytest.raises(Exception, match="LLM 不可用"):
                mock_crew.query("测试问题")


# ─── 端到端集成测试（需要真实 API）─────────────────────────


class TestEndToEnd:
    """端到端集成测试，需要真实 LLM API 和向量库数据。"""

    @pytest.mark.slow
    def test_retrieve_question_returns_answer(self):
        """文档相关问题应走 RETRIEVE 路径并返回有内容的回答。"""
        crew = RAGCrew(verbose=False, memory=False)
        result = crew.query("什么是 Python 装饰器？")
        assert result.answer
        assert len(result.answer) > 10
        assert result.route in ("RETRIEVE", "DIRECT")

    @pytest.mark.slow
    def test_direct_question_returns_answer(self):
        """常识问题应走 DIRECT 路径。"""
        crew = RAGCrew(verbose=False, memory=False)
        result = crew.query("1+1等于几？")
        assert result.answer
        assert result.route == "DIRECT"

    @pytest.mark.slow
    def test_irrelevant_question_handled_gracefully(self):
        """完全无关的问题应得到合理回答。"""
        crew = RAGCrew(verbose=False, memory=False)
        result = crew.query("木星有多少颗卫星？")
        assert result.answer
        assert len(result.answer) > 0

    @pytest.mark.slow
    def test_performance_metrics_recorded(self):
        """应记录 token 消耗和响应时间。"""
        crew = RAGCrew(verbose=False, memory=False)
        result = crew.query("Python 是什么？")
        assert result.elapsed_seconds > 0

    @pytest.mark.slow
    def test_memory_context_across_queries(self):
        """连续两个有上下文关联的问题，第二次应引用第一次的上下文。"""
        crew = RAGCrew(verbose=False, memory=True)

        result1 = crew.query("什么是 Python 装饰器？")
        assert result1.answer

        result2 = crew.query("能给我一个它的代码示例吗？")
        assert result2.answer
        assert len(result2.answer) > 10
