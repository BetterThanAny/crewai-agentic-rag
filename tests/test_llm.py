"""M1 测试：LLM 连通性与配置管理。"""

import importlib
import os
from unittest.mock import patch

import pytest


class TestConfig:
    """配置管理测试。"""

    def _reload_config(self):
        """重新加载 config 模块以获取最新环境变量。"""
        import src.config as config_module

        importlib.reload(config_module)
        return config_module

    def test_missing_api_key_raises_error(self):
        """LLM_API_KEY 缺失时应抛出 ValueError。"""
        with patch.dict(os.environ, {"LLM_PROVIDER": "deepseek"}, clear=False):
            config = self._reload_config()
            # reload 后再清除 key，避免 load_dotenv 从 .env 重新读入
            os.environ.pop("LLM_API_KEY", None)
            with pytest.raises(ValueError, match="LLM_API_KEY"):
                config.get_llm()

    def test_ollama_no_api_key_required(self):
        """Ollama provider 不需要 API key。"""
        env = {"LLM_PROVIDER": "ollama", "LLM_MODEL": "qwen2.5:7b"}
        with patch.dict(os.environ, env, clear=False):
            config = self._reload_config()
            llm = config.get_llm()
            assert llm is not None

    def test_default_provider_is_deepseek(self):
        """默认 provider 应该是 deepseek。"""
        env = {"LLM_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=False):
            # 移除 provider 让它走默认值
            os.environ.pop("LLM_PROVIDER", None)
            config = self._reload_config()
            llm = config.get_llm()
            assert "deepseek" in llm.model or "openai" in llm.model

    def test_embedding_config_returns_dict(self):
        """get_embedding_config 应返回包含 provider 和 config 的字典。"""
        config = self._reload_config()
        result = config.get_embedding_config()
        assert "provider" in result
        assert "config" in result
        assert "model_name" in result["config"]

    def test_qwen_provider(self):
        """Qwen provider 应正确配置。"""
        env = {
            "LLM_PROVIDER": "qwen",
            "LLM_MODEL": "qwen-plus",
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        with patch.dict(os.environ, env, clear=False):
            config = self._reload_config()
            llm = config.get_llm()
            assert "qwen" in llm.model


class TestLLMConnectivity:
    """LLM 连通性测试（需要真实 API key，标记为 slow）。"""

    @pytest.mark.slow
    def test_llm_responds(self):
        """LLM API 应正常返回响应。"""
        from src.config import get_llm

        llm = get_llm()
        response = llm.call(messages=[{"role": "user", "content": "回复OK两个字"}])
        assert response is not None
        assert len(str(response)) > 0

    @pytest.mark.slow
    def test_llm_handles_chinese(self):
        """LLM 应正确处理中文输入输出。"""
        from src.config import get_llm

        llm = get_llm()
        response = llm.call(
            messages=[{"role": "user", "content": "用中文回复：1+1等于几？"}]
        )
        assert "2" in str(response) or "二" in str(response)
