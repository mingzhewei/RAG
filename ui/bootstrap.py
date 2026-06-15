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
from sensor_vector_db.core.import_jobs import ImportJobManager  # noqa: E402
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
    apply_compact_style()
    settings = get_settings()
    configure_logging(settings)
    init_database(settings)


def apply_compact_style() -> None:
    """Apply compact Streamlit layout defaults for dense technical tables."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 1.5rem;
            max-width: 100%;
        }
        html, body, [data-testid="stAppViewContainer"] {
            font-size: 14px;
        }
        h1 { font-size: 1.55rem; margin-bottom: 0.7rem; }
        h2 { font-size: 1.2rem; margin-top: 1rem; }
        h3 { font-size: 1.02rem; }
        p, li, label, [data-testid="stMarkdownContainer"] {
            line-height: 1.35;
        }
        [data-testid="stHorizontalBlock"] {
            gap: 0.55rem;
        }
        [data-testid="stMetric"] {
            min-width: 0;
            overflow: hidden;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 0.76rem;
            white-space: normal;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.05rem;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        [data-testid="stCaptionContainer"] {
            font-size: 0.76rem;
            line-height: 1.28;
            overflow-wrap: anywhere;
        }
        .stDataFrame {
            font-size: 0.78rem;
        }
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            min-height: 2rem;
            padding: 0.25rem 0.65rem;
            white-space: normal;
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-baseweb="select"] {
            font-size: 0.86rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_runtime_llm_settings(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    wire_api: str | None = None,
) -> None:
    """Apply LLM settings for the current Streamlit process."""
    os.environ["LLM_PROVIDER"] = provider.strip().lower()
    if wire_api and wire_api.strip():
        os.environ["WIRE_API"] = wire_api.strip().lower()
    if base_url and base_url.strip():
        if provider == "crs":
            os.environ["CRS_BASE_URL"] = base_url.strip()
        elif provider == "deepseek":
            os.environ["DEEPSEEK_BASE_URL"] = base_url.strip()
    if model and model.strip():
        if provider == "crs":
            os.environ["CRS"] = model.strip()
        elif provider == "deepseek":
            os.environ["DEEPSEEK_MODEL"] = model.strip()
    if api_key and api_key.strip():
        if provider == "crs":
            os.environ["CRS_API_KEY"] = api_key.strip()
        elif provider == "deepseek":
            os.environ["DEEPSEEK_API_KEY"] = api_key.strip()
    get_settings.cache_clear()


def apply_runtime_api_key(api_key: str) -> None:
    """Apply an API key for the current selected LLM provider."""
    settings = get_settings()
    apply_runtime_llm_settings(settings.llm_provider, api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_document_manager() -> DocumentManager:
    """Return cached document manager."""
    return DocumentManager(get_settings())


@st.cache_resource(show_spinner=False)
def get_import_job_manager() -> ImportJobManager:
    """Return cached import job manager."""
    return ImportJobManager(get_settings())


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
    get_import_job_manager.clear()
    get_search_engine.clear()
    get_qa_system.clear()
    get_parameter_extractor.clear()
    get_parameter_comparer.clear()
