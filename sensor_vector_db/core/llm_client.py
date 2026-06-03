"""DeepSeek chat client for source-grounded generation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sensor_vector_db.config.settings import Settings, get_settings


class DeepseekChatClient:
    """OpenAI-compatible DeepSeek chat client."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the client lazily."""
        self.settings = settings or get_settings()
        self._client = None
        self.call_count = 0

    def _load_client(self):
        """Load the OpenAI SDK client configured for DeepSeek."""
        if self._client is not None:
            return self._client
        if not self.settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for DeepSeek calls.") from exc
        self._client = OpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
            timeout=self.settings.deepseek_timeout_seconds,
        )
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        stream: bool = False,
    ) -> str | Iterable[str]:
        """Call DeepSeek chat completions."""
        client = self._load_client()
        try:
            self.call_count += 1
            response = client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )
            if stream:
                return self._stream_text(response)
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError(f"DeepSeek chat call failed: {exc}") from exc

    @staticmethod
    def _stream_text(response: Any) -> Iterable[str]:
        """Yield text deltas from a streaming response."""
        for chunk in response:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content


class NullLLMClient:
    """Local fallback client used when no DeepSeek token is configured."""

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        stream: bool = False,
    ) -> str:
        """Return a deterministic source-grounded fallback response."""
        del temperature, stream
        user_message = messages[-1]["content"] if messages else ""
        return (
            "未配置 DEEPSEEK_API_KEY，无法调用 DeepSeek 生成自然语言回答。\n\n"
            "已完成本地检索，请根据下方来源片段核对：\n"
            f"{user_message[:1200]}"
        )

