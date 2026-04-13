"""M2b 测试：Tool 单元测试。"""

from unittest.mock import patch

from src.tools.vector_search_tool import _mock_search, vector_search_tool


class TestMockSearch:
    """Mock 检索函数测试。"""

    def test_returns_list_of_strings(self):
        """_mock_search 应返回字符串列表。"""
        results = _mock_search("测试查询")
        assert isinstance(results, list)
        assert all(isinstance(r, str) for r in results)

    def test_default_returns_3_results(self):
        """默认返回 3 条结果。"""
        results = _mock_search("测试查询")
        assert len(results) == 3

    def test_top_k_limits_results(self):
        """top_k 参数应限制返回数量。"""
        results = _mock_search("测试查询", top_k=1)
        assert len(results) == 1

    def test_results_contain_query(self):
        """结果中应包含查询关键词。"""
        query = "人工智能"
        results = _mock_search(query)
        assert all(query in r for r in results)


class TestVectorSearchTool:
    """vector_search_tool CrewAI Tool 测试。"""

    def test_tool_has_name(self):
        """Tool 应有正确的名称。"""
        assert vector_search_tool.name == "vector_search_tool"

    def test_tool_has_description(self):
        """Tool 应有描述。"""
        assert vector_search_tool.description
        assert len(vector_search_tool.description) > 10

    def test_tool_returns_string(self):
        """Tool 执行结果应为字符串。"""
        result = vector_search_tool.run(query="测试查询")
        assert isinstance(result, str)

    def test_tool_result_contains_separator(self):
        """多条结果应以分隔符连接。"""
        result = vector_search_tool.run(query="测试查询", top_k=3)
        assert "---" in result

    def test_tool_single_result(self):
        """top_k=1 时应只返回一条结果。"""
        result = vector_search_tool.run(query="测试查询", top_k=1)
        assert "---" not in result

    @patch("src.tools.vector_search_tool._try_import_vector_store", return_value=None)
    def test_fallback_to_mock_when_no_vector_store(self, mock_import):
        """M2a 未就绪时应回退到 mock 数据。"""
        result = vector_search_tool.run(query="测试查询")
        assert "Mock 结果" in result
