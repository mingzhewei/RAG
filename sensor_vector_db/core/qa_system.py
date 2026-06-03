"""Source-grounded RAG question answering."""

from __future__ import annotations

import json

from sensor_vector_db.config.settings import Settings, get_settings
from sensor_vector_db.core.llm_client import DeepseekChatClient, NullLLMClient
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
        llm_client: DeepseekChatClient | NullLLMClient | None = None,
    ) -> None:
        """Initialize QA system."""
        self.settings = settings or get_settings()
        self.search_engine = search_engine or SearchEngine(self.settings)
        self.llm_client = llm_client or (
            DeepseekChatClient(self.settings)
            if self.settings.deepseek_api_key
            else NullLLMClient()
        )

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
        messages = [
            {
                "role": "system",
                "content": (
                    "你是专业传感器技术文档检索问答助手。必须严格基于用户提供的"
                    "【来源片段】回答。禁止使用来源片段之外的知识补充事实。"
                    "如果来源不足，明确说“未在已入库文档中找到依据”。"
                    "每条关键结论后必须标注来源编号，例如 [S1]。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n【来源片段】\n{context}",
            },
        ]
        try:
            answer = str(self.llm_client.chat(messages, temperature=0.0))
        except Exception as exc:
            answer = f"DeepSeek 调用失败：{exc}\n\n已返回本地检索来源，请人工核对。"
        self._save_history(question, answer, results)
        return {"answer": answer, "sources": results}

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

