"""Source-grounded RAG question answering."""

from __future__ import annotations

import json
from collections.abc import Generator

from sensor_vector_db.config.settings import Settings, get_settings
from sensor_vector_db.core.import_jobs import classify_error
from sensor_vector_db.core.llm_client import (
    NullLLMClient,
    OpenAICompatibleChatClient,
    create_llm_client,
)
from sensor_vector_db.core.search_engine import SearchEngine
from sensor_vector_db.core.types import SearchResult
from sensor_vector_db.models.database import QueryHistory, session_scope


NO_EVIDENCE_MESSAGE = "未在已入库文档中找到依据。"


class QASystem:
    """RAG system that answers only from retrieved evidence."""

    def __init__(
        self,
        settings: Settings | None = None,
        search_engine: SearchEngine | None = None,
        llm_client: OpenAICompatibleChatClient | NullLLMClient | None = None,
    ) -> None:
        """Initialize QA system."""
        self.settings = settings or get_settings()
        self.search_engine = search_engine or SearchEngine(self.settings)
        self.llm_client = llm_client or create_llm_client(self.settings)

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> dict:
        """Answer a question from retrieved document chunks."""
        results = self.search_engine.search(
            question,
            mode="hybrid",
            top_k=top_k or self.settings.search_top_k,
            filters=filters,
        )
        if not results:
            answer = NO_EVIDENCE_MESSAGE
            self._save_history(question, answer, [])
            return {"answer": answer, "sources": []}

        context = self._format_context(results)
        messages = self._build_messages(question, context)
        try:
            answer = str(self.llm_client.chat(messages, temperature=0.0))
        except Exception as exc:
            answer = f"{classify_error(exc, '大模型问答')}\n\n已返回本地检索来源，请人工核对。"
        self._save_history(question, answer, results)
        return {"answer": answer, "sources": results}

    def answer_stream(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> Generator[str, None, dict]:
        """Answer a question with streaming output.

        Yields text chunks as they arrive, and returns the final dict
        (answer + sources) via Generator return value.
        """
        results = self.search_engine.search(
            question,
            mode="hybrid",
            top_k=top_k or self.settings.search_top_k,
            filters=filters,
        )
        if not results:
            yield NO_EVIDENCE_MESSAGE
            self._save_history(question, NO_EVIDENCE_MESSAGE, [])
            return {"answer": NO_EVIDENCE_MESSAGE, "sources": []}

        context = self._format_context(results)
        messages = self._build_messages(question, context)

        full_answer = ""
        try:
            for chunk in self.llm_client.chat(messages, temperature=0.0, stream=True):
                if isinstance(chunk, str):
                    full_answer += chunk
                    yield chunk
        except Exception as exc:
            error_msg = f"\n\n{classify_error(exc, '大模型问答')}\n\n已返回本地检索来源，请人工核对。"
            full_answer += error_msg
            yield error_msg

        if not full_answer:
            full_answer = "（模型未返回内容）"
            yield full_answer

        self._save_history(question, full_answer, results)
        return {"answer": full_answer, "sources": results}

    @staticmethod
    def _build_messages(question: str, context: str) -> list[dict[str, str]]:
        """Build chat messages for the LLM."""
        return [
            {
                "role": "system",
                "content": (
                    "你是专业传感器技术文档检索问答助手。必须严格基于用户提供的"
                    "【来源片段】回答。禁止使用来源片段之外的知识补充事实。"
                    '如果来源不足，明确说"未在已入库文档中找到依据"。'
                    "每条关键结论后必须标注来源编号，例如 [S1]。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n【来源片段】\n{context}",
            },
        ]

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format retrieved chunks for the LLM prompt."""
        parts = []
        for index, result in enumerate(results, start=1):
            page = f" p.{result.page_number}" if result.page_number else ""
            parts.append(
                f"[S{index}] {result.source}{page}\n"
                f"chunk_id={result.chunk_id}\n"
                f"{result.content[:1800]}"
            )
        return "\n\n".join(parts)

    def _save_history(
        self,
        question: str,
        answer: str,
        sources: list[SearchResult],
    ) -> None:
        """Persist query history with source identifiers."""
        source_payload = [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "source": item.source,
                "page_number": item.page_number,
                "score": item.score,
            }
            for item in sources
        ]
        with session_scope(self.settings) as session:
            session.add(
                QueryHistory(
                    query=question,
                    query_type="qa",
                    answer=answer,
                    sources_json=json.dumps(source_payload, ensure_ascii=False),
                )
            )
