"""智谱大模型调用示例，同时为其他模块提供统一的调用函数。"""

import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().with_name(".env"))


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}，请在项目 .env 文件中配置")
    return value


LLM_URL = _required_env("LLM_URL")
LLM_MODEL = _required_env("LLM_MODEL")
LLM_API_KEY = _required_env("LLM_API_KEY")


def call_llm(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    reasoning_effort: str = "max",
    timeout: int = 180,
) -> str:
    """调用大模型并返回 assistant 的文本内容。"""
    payload: dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "top_p": 0.95,
        "reasoning_effort": reasoning_effort,
    }
    response = requests.post(
        LLM_URL,
        json=payload,
        headers={
            "Authorization": LLM_API_KEY,
            "Content-Type": "application/json",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"大模型返回格式异常：{data}") from exc


if __name__ == "__main__":
    result = call_llm(
        [
            {"role": "system", "content": "你是编程助手，擅长写简洁高效的代码。"},
            {"role": "user", "content": "写一个 Python 函数，计算斐波那契数列第 n 项。"},
        ],
        temperature=1,
    )
    print(result)
