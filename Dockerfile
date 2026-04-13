FROM python:3.12-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 复制依赖文件并安装
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 复制项目代码
COPY src/ src/
COPY data/ data/
COPY app.py main.py ingest.py ./

# Streamlit 配置
RUN mkdir -p .streamlit
COPY .streamlit/config.toml .streamlit/config.toml

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["uv", "run", "streamlit", "run", "app.py", \
    "--server.port=8501", "--server.address=0.0.0.0"]
