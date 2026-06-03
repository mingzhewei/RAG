"""Evidence-backed sensor parameter extraction and comparison."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import json
import re
from typing import Any

from sqlalchemy import delete, select

from sensor_vector_db.config.settings import Settings, get_settings
from sensor_vector_db.core.llm_client import DeepseekChatClient
from sensor_vector_db.models.database import (
    Document,
    DocumentChunk,
    ExtractedParameter,
    session_scope,
)
from sensor_vector_db.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ParameterValue:
    """One extracted parameter value with source evidence."""

    name: str
    normalized_name: str
    value: str
    unit: str | None
    source_text: str
    page_number: int | None
    confidence: float = 0.75


@dataclass
class ComparisonCell:
    """One parameter comparison table cell."""

    value: str
    source: str


class ParameterExtractor:
    """Extract sensor parameters from stored chunks without inventing values."""

    FIELD_ALIASES: dict[str, tuple[str, ...]] = {
        "manufacturer": ("制造商", "厂商", "品牌", "Manufacturer", "Brand"),
        "model": ("型号", "产品型号", "Model", "Part Number", "P/N"),
        "sensor_type": ("类型", "传感器类型", "Sensor Type", "Type"),
        "range": ("量程", "测量范围", "测距范围", "Range", "Measurement Range"),
        "accuracy": ("精度", "准确度", "Accuracy"),
        "resolution": ("分辨率", "Resolution"),
        "sampling_rate": ("采样率", "刷新率", "Sampling Rate", "Rate"),
        "response_time": ("响应时间", "Response Time"),
        "operating_temperature": ("工作温度", "Operating Temperature"),
        "power": ("电源", "供电", "Power", "Supply Voltage"),
        "interface": ("接口", "通信接口", "Interface", "Communication"),
        "dimensions": ("尺寸", "外形尺寸", "Dimensions", "Size"),
        "weight": ("重量", "Weight"),
        "ip_rating": ("防护等级", "IP 等级", "IP Rating", "Ingress Protection"),
        "fov": ("视场角", "FOV", "Field of View"),
        "wavelength": ("波长", "Wavelength"),
    }

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize extractor."""
        self.settings = settings or get_settings()
        self.llm_client = DeepseekChatClient(self.settings)

    def extract_for_document(self, document_id: str, use_llm: bool = True) -> list[ParameterValue]:
        """Extract and persist parameters for one document."""
        with session_scope(self.settings) as session:
            document = session.get(Document, document_id)
            if not document:
                raise ValueError(f"Document not found: {document_id}")
            chunks = session.execute(
                select(DocumentChunk).where(DocumentChunk.document_id == document_id)
            ).scalars().all()
            session.execute(
                delete(ExtractedParameter).where(ExtractedParameter.document_id == document_id)
            )
            parameters = self.extract_from_chunks(chunks)
            if use_llm and self.settings.deepseek_api_key:
                parameters = self._llm_validate(parameters, chunks)
            sensor_model = document.sensor_model or _first_value(parameters, "model")
            for parameter in parameters:
                session.add(
                    ExtractedParameter(
                        document_id=document_id,
                        sensor_model=sensor_model,
                        name=parameter.name,
                        normalized_name=parameter.normalized_name,
                        value=parameter.value,
                        unit=parameter.unit,
                        source_text=parameter.source_text,
                        page_number=parameter.page_number,
                        confidence=parameter.confidence,
                    )
                )
            if not document.sensor_model:
                document.sensor_model = sensor_model
            if not document.manufacturer:
                document.manufacturer = _first_value(parameters, "manufacturer")
            return parameters

    def extract_from_chunks(self, chunks: list[DocumentChunk]) -> list[ParameterValue]:
        """Extract parameters from chunk objects using rules."""
        extracted: list[ParameterValue] = []
        seen: set[tuple[str, str, str | None]] = set()
        for chunk in chunks:
            for parameter in self._extract_from_text(chunk.content, chunk.page_number):
                key = (
                    parameter.normalized_name,
                    parameter.value.lower(),
                    parameter.page_number,
                )
                if key not in seen:
                    extracted.append(parameter)
                    seen.add(key)
        return extracted

    def _extract_from_text(self, text: str, page_number: int | None) -> list[ParameterValue]:
        """Extract parameters from one text chunk."""
        parameters = []
        parameters.extend(self._extract_markdown_table(text, page_number))
        parameters.extend(self._extract_key_value_lines(text, page_number))
        return parameters

    def _extract_markdown_table(
        self,
        text: str,
        page_number: int | None,
    ) -> list[ParameterValue]:
        """Extract key-value pairs from Markdown tables."""
        lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
        if len(lines) < 2:
            return []
        parameters: list[ParameterValue] = []
        for line in lines:
            if "---" in line:
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            name, value = cells[0], cells[1]
            if not name or not value:
                continue
            normalized = self.normalize_name(name)
            if normalized:
                clean_value, unit = split_value_unit(value)
                parameters.append(
                    ParameterValue(
                        name=name,
                        normalized_name=normalized,
                        value=clean_value,
                        unit=unit,
                        source_text=line,
                        page_number=page_number,
                        confidence=0.85,
                    )
                )
        return parameters

    def _extract_key_value_lines(
        self,
        text: str,
        page_number: int | None,
    ) -> list[ParameterValue]:
        """Extract parameter values from key-value style lines."""
        parameters: list[ParameterValue] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if len(line) < 4 or len(line) > 300:
                continue
            match = re.match(r"^([^:：]{2,60})\s*[:：]\s*(.{1,180})$", line)
            if not match:
                continue
            name = match.group(1).strip()
            value = match.group(2).strip(" ;,，。")
            normalized = self.normalize_name(name)
            if not normalized or not value:
                continue
            clean_value, unit = split_value_unit(value)
            parameters.append(
                ParameterValue(
                    name=name,
                    normalized_name=normalized,
                    value=clean_value,
                    unit=unit,
                    source_text=line,
                    page_number=page_number,
                    confidence=0.8,
                )
            )
        return parameters

    def normalize_name(self, name: str) -> str | None:
        """Map a parameter label to a normalized field name."""
        normalized_text = re.sub(r"\s+", " ", name.strip()).lower()
        for normalized, aliases in self.FIELD_ALIASES.items():
            for alias in aliases:
                if alias.lower() in normalized_text:
                    return normalized
        return None

    def _llm_validate(
        self,
        parameters: list[ParameterValue],
        chunks: list[DocumentChunk],
    ) -> list[ParameterValue]:
        """Ask DeepSeek to validate field names, never to invent missing values."""
        if not parameters:
            return parameters
        sample = [
            {
                "name": item.name,
                "normalized_name": item.normalized_name,
                "value": item.value,
                "unit": item.unit,
                "source_text": item.source_text,
                "page_number": item.page_number,
            }
            for item in parameters[:80]
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "你是严谨的传感器资料参数校验助手。只能校验和规范已有候选值，"
                    "禁止新增候选中不存在的参数。输出 JSON 数组。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(sample, ensure_ascii=False),
            },
        ]
        try:
            response = self.llm_client.chat(messages)
            decoded = json.loads(_extract_json(str(response)))
        except Exception as exc:
            logger.warning("LLM parameter validation skipped: %s", exc)
            return parameters
        valid_keys = {
            (item.name, item.value, item.page_number): item
            for item in parameters
        }
        validated: list[ParameterValue] = []
        for item in decoded if isinstance(decoded, list) else []:
            key = (item.get("name"), item.get("value"), item.get("page_number"))
            original = valid_keys.get(key)
            if not original:
                continue
            normalized = str(item.get("normalized_name") or original.normalized_name)
            validated.append(
                ParameterValue(
                    name=original.name,
                    normalized_name=normalized,
                    value=original.value,
                    unit=original.unit,
                    source_text=original.source_text,
                    page_number=original.page_number,
                    confidence=min(1.0, original.confidence + 0.05),
                )
            )
        return validated or parameters


class ParameterComparer:
    """Compare evidence-backed parameters across 2-5 sensor models."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize comparer."""
        self.settings = settings or get_settings()

    def compare_models(self, models: list[str]) -> dict[str, dict[str, ComparisonCell]]:
        """Build a normalized parameter comparison table."""
        clean_models = [model.strip() for model in models if model.strip()]
        if len(clean_models) < 2 or len(clean_models) > 5:
            raise ValueError("Parameter comparison requires 2-5 sensor models.")
        with session_scope(self.settings) as session:
            rows = session.execute(
                select(ExtractedParameter, Document)
                .join(Document, Document.id == ExtractedParameter.document_id)
                .where(ExtractedParameter.sensor_model.in_(clean_models))
            ).all()
            table: dict[str, dict[str, ComparisonCell]] = {}
            for parameter, document in rows:
                model = parameter.sensor_model or document.sensor_model or "未识别型号"
                source = f"{document.filename}"
                if parameter.page_number:
                    source += f" p.{parameter.page_number}"
                source += f" / {parameter.source_text[:80]}"
                table.setdefault(parameter.normalized_name, {})
                if model not in table[parameter.normalized_name]:
                    value = parameter.value
                    if parameter.unit and parameter.unit not in value:
                        value = f"{value} {parameter.unit}"
                    table[parameter.normalized_name][model] = ComparisonCell(value, source)
            for parameter_name in list(table.keys()):
                for model in clean_models:
                    table[parameter_name].setdefault(
                        model,
                        ComparisonCell("未找到", "未在已入库文档中找到依据"),
                    )
            return table

    def to_markdown(
        self,
        table: dict[str, dict[str, ComparisonCell]],
        models: list[str],
    ) -> str:
        """Render comparison table as Markdown with source in each cell."""
        header = ["参数", *models]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        for parameter_name in sorted(table):
            row = [parameter_name]
            for model in models:
                cell = table[parameter_name].get(
                    model,
                    ComparisonCell("未找到", "未在已入库文档中找到依据"),
                )
                row.append(f"{cell.value}<br><small>{cell.source}</small>")
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def to_csv(
        self,
        table: dict[str, dict[str, ComparisonCell]],
        models: list[str],
    ) -> str:
        """Render comparison table as CSV with value and source per cell."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["参数", *models])
        for parameter_name in sorted(table):
            row = [parameter_name]
            for model in models:
                cell = table[parameter_name].get(
                    model,
                    ComparisonCell("未找到", "未在已入库文档中找到依据"),
                )
                row.append(f"{cell.value} | 来源: {cell.source}")
            writer.writerow(row)
        return output.getvalue()


def split_value_unit(value: str) -> tuple[str, str | None]:
    """Split a value into value text and likely unit without guessing."""
    cleaned = value.strip()
    match = re.match(r"^([<>≤≥~±+\-]?\s*[\d.]+(?:\s*[-~]\s*[\d.]+)?)(\s*[^\d\s]+.*)?$", cleaned)
    if match:
        raw_value = match.group(1).strip()
        unit = (match.group(2) or "").strip() or None
        return raw_value, unit
    return cleaned, None


def _first_value(parameters: list[ParameterValue], normalized_name: str) -> str | None:
    """Return the first value for a normalized parameter name."""
    for parameter in parameters:
        if parameter.normalized_name == normalized_name:
            return parameter.value
    return None


def _extract_json(text: str) -> str:
    """Extract the first JSON array/object substring from text."""
    match = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
    return match.group(1) if match else text

