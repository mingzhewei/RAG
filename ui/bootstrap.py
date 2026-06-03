"""Streamlit bootstrap helpers."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from sensor_vector_db.config.settings import get_settings  # noqa: E402
from sensor_vector_db.core.document_manager import DocumentManager  # noqa: E402
from sensor_vector_db.core.parameter_extractor import (  # noqa: E402
    ParameterComparer,
    ParameterExtractor,
)
from sensor_vector_db.core.qa_system import QASystem  # noqa: E402
from sensor_vector_db.core.search_engine import SearchEngine  # noqa: E402
from sensor_vector_db.models.database import init_database  # noqa: E402
from sensor_vector_db.utils.logger import configure_logging  # noqa: E402


def configure_page(title: str) -> None:
    """Configure a Streamlit page and core services."""
    st.set_page_config(page_title=title, page_icon="🔎", layout="wide")
    settings = get_settings()
    configure_logging(settings)
    init_database(settings)


def apply_runtime_api_key(api_key: str) -> None:
    """Apply a DeepSeek API key for the current Streamlit process."""
    if api_key.strip():
        os.environ["DEEPSEEK_API_KEY"] = api_key.strip()
        get_settings.cache_clear()


@st.cache_resource(show_spinner=False)
def get_document_manager() -> DocumentManager:
    """Return cached document manager."""
    return DocumentManager(get_settings())


@st.cache_resource(show_spinner=False)
def get_search_engine() -> SearchEngine:
    """Return cached search engine."""
    return SearchEngine(get_settings())


@st.cache_resource(show_spinner=False)
def get_qa_system() -> QASystem:
    """Return cached QA system."""
    return QASystem(get_settings())


@st.cache_resource(show_spinner=False)
def get_parameter_extractor() -> ParameterExtractor:
    """Return cached parameter extractor."""
    return ParameterExtractor(get_settings())


@st.cache_resource(show_spinner=False)
def get_parameter_comparer() -> ParameterComparer:
    """Return cached parameter comparer."""
    return ParameterComparer(get_settings())


def clear_resource_caches() -> None:
    """Clear cached resources after settings changes."""
    get_document_manager.clear()
    get_search_engine.clear()
    get_qa_system.clear()
    get_parameter_extractor.clear()
    get_parameter_comparer.clear()

