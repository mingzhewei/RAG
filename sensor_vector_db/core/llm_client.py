"""OpenAI-compatible chat clients for source-grounded generation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sensor_vector_db.config.settings import Settings, get_settings


class OpenAICompatibleChatClient:
    """OpenAI SDK client configured from the selected provider settings."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the client lazily."""
        self.settings = settings or get_settings()
        self._client = None
        self.call_count = 0

    def _load_client(self):
        """Load an OpenAI SDK client for the active OpenAI-compatible endpoint."""
        if self._client is not None:
            return self._client
        if not self.settings.active_llm_api_key:
            raise RuntimeError(f"{self.settings.llm_provider.upper()} API key is not configured.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for LLM calls.") from exc
        self._client = OpenAI(
            api_key=self.settings.active_llm_api_key,
            base_url=self.settings.active_llm_base_url,
            timeout=self.settings.llm_timeout_seconds,
        )
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        stream: bool = False,
    ) -> str | Iterable[str]:
        """Call the selected OpenAI-compatible API surface."""
        client = self._load_client()
        try:
            self.call_count += 1
            if self.settings.wire_api == "responses":
                response = client.responses.create(
                    model=self.settings.active_llm_model,
                    input=self._responses_input(messages),
                    instructions=self._responses_instructions(messages),
                    temperature=temperature,
                    stream=stream,
                )
                if stream:
                    return self._stream_responses_text(response)
                return self._response_text(response)
            response = client.chat.completions.create(
                model=self.settings.active_llm_model,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )
            if stream:
                return self._stream_chat_text(response)
            return response.choices[0].message.content or ""
        except Exception as exc:
            provider = self.settings.llm_provider.upper()
            raise RuntimeError(f"{provider} LLM call failed: {exc}") from exc

    @staticmethod
    def _responses_instructions(messages: list[dict[str, str]]) -> str | None:
        """Build Responses API instructions from system messages."""
        instructions = [
            message.get("content", "")
            for message in messages
            if message.get("role") == "system" and message.get("content")
        ]
        return "\n\n".join(instructions) if instructions else None

    @staticmethod
    def _responses_input(messages: list[dict[str, str]]) -> str:
        """Build a provider-compatible text input for Responses API calls."""
        parts = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                continue
            content = message.get("content", "")
            if content:
                parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    @staticmethod
    def _response_text(response: Any) -> str:
        """Extract plain text from a Responses API result."""
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)
        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)

    @staticmethod
    def _stream_chat_text(response: Any) -> Iterable[str]:
        """Yield text deltas from a Chat Completions stream."""
        for chunk in response:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content

    @staticmethod
    def _stream_responses_text(response: Any) -> Iterable[str]:
        """Yield text deltas from a Responses API stream."""
        for event in response:
            if getattr(event, "type", "") == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if delta:
                    yield str(delta)


class DeepseekChatClient(OpenAICompatibleChatClient):
    """Backward-compatible DeepSeek client wrapper."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize with DeepSeek provider regardless of the global default."""
        resolved = settings or get_settings()
        super().__init__(resolved.model_copy(update={"llm_provider": "deepseek"}))


def create_llm_client(settings: Settings | None = None) -> OpenAICompatibleChatClient | NullLLMClient:
    """Create the active LLM client or a local fallback when disabled/unconfigured."""
    resolved = settings or get_settings()
    if resolved.llm_provider == "none" or not resolved.active_llm_api_key:
        return NullLLMClient(resolved)
    return OpenAICompatibleChatClient(resolved)


class NullLLMClient:
    """Local fallback client used when no provider key is configured."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize fallback with current settings for user-facing messages."""
        self.settings = settings or get_settings()

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
            f"未配置可用的大模型提供方（当前 LLM_PROVIDER={self.settings.llm_provider}），"
            "无法生成自然语言回答。\n\n"
            "已完成本地检索，请根据下方来源片段核对：\n"
            f"{user_message[:1200]}"
        )
