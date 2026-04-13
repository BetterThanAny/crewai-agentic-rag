"""LLM 与 Embedding 统一配置模块。

从 .env 文件读取配置，提供 get_llm() 和 get_embedding_config() 两个核心函数。
支持 4 种 provider：DeepSeek、Qwen、OpenAI 兼容中转站、本地 Ollama。
"""

import os

from crewai import LLM
from dotenv import load_dotenv

load_dotenv()


def get_llm() -> LLM:
    """返回配置好的 CrewAI LLM 实例。

    根据 LLM_PROVIDER 环境变量选择对应 provider：
    - deepseek / qwen / openai_proxy：需要 LLM_API_KEY
    - ollama：本地模式，无需 API key

    Raises:
        ValueError: 非 ollama provider 下缺少 LLM_API_KEY
    """
    provider = os.getenv("LLM_PROVIDER", "deepseek")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")

    if provider == "ollama":
        return LLM(
            model=model,
            provider="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    if not api_key:
        raise ValueError(
            f"LLM_API_KEY is required for provider: {provider}. "
            "请在 .env 文件中设置 LLM_API_KEY。"
        )

    # 映射到 CrewAI 原生 provider
    # deepseek → deepseek, qwen → dashscope, openai_proxy → openai
    provider_map = {
        "deepseek": "deepseek",
        "qwen": "dashscope",
        "openai_proxy": "openai",
    }
    crewai_provider = provider_map.get(provider, "openai")

    kwargs: dict = {
        "model": model,
        "provider": crewai_provider,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return LLM(**kwargs)


def get_embedding_config() -> dict:
    """返回 Embedding 配置字典，供向量库使用。

    Returns:
        包含 provider 和 config 的字典，格式兼容 CrewAI embedding 配置。
    """
    return {
        "provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
        "config": {
            "model_name": os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
            "api_key": os.getenv("EMBEDDING_API_KEY"),
            "api_base": os.getenv("EMBEDDING_BASE_URL"),
        },
    }
