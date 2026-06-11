"""SQLite metadata store for documents, chunks, parameters, and history."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
import uuid

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from sqlalchemy.orm import sessionmaker

from sensor_vector_db.config.settings import Settings, get_settings


class Base(DeclarativeBase):
    """Base SQLAlchemy model class."""


def new_id() -> str:
    """Generate a stable string UUID for database records."""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Return a naive UTC timestamp for SQLite compatibility."""
    return datetime.now(UTC).replace(tzinfo=None)


class Document(Base):
    """Imported source document metadata."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    file_path: Mapped[str] = mapped_column(String(2048), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(512), index=True)
    file_type: Mapped[str] = mapped_column(String(64), index=True)
    file_hash: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    modified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    status: Mapped[str] = mapped_column(String(32), default="imported", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    manufacturer: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    sensor_model: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    index_profile: Mapped[str | None] = mapped_column(Text, nullable=True)

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
    parameters: Mapped[list["ExtractedParameter"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    """Indexed text chunk with source metadata."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, index=True)
    content: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(String(64), default="text")
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_label: Mapped[str] = mapped_column(String(512))
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    document: Mapped[Document] = relationship(back_populates="chunks")


class ExtractedParameter(Base):
    """Sensor parameter extracted from source-backed text."""

    __tablename__ = "extracted_parameters"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    sensor_model: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(256), index=True)
    normalized_name: Mapped[str] = mapped_column(String(256), index=True)
    value: Mapped[str] = mapped_column(Text)
    unit: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_text: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    document: Mapped[Document] = relationship(back_populates="parameters")


class QueryHistory(Base):
    """User query history for audit and reproducibility."""

    __tablename__ = "query_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    query: Mapped[str] = mapped_column(Text)
    query_type: Mapped[str] = mapped_column(String(64), index=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    sources_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class ImportJob(Base):
    """Persistent background document import job."""

    __tablename__ = "import_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    source_path: Mapped[str] = mapped_column(String(2048))
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    phase: Mapped[str] = mapped_column(String(128), default="等待开始")
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_file: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    current_index: Mapped[int] = mapped_column(Integer, default=0)
    imported: Mapped[int] = mapped_column(Integer, default=0)
    updated: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0)
    deleted: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    report_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    events: Mapped[list["ImportJobEvent"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )
    files: Mapped[list["ImportJobFile"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )


class ImportJobEvent(Base):
    """One timestamped progress event for an import job."""

    __tablename__ = "import_job_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("import_jobs.id", ondelete="CASCADE"),
        index=True,
    )
    level: Mapped[str] = mapped_column(String(32), default="info")
    phase: Mapped[str] = mapped_column(String(128), default="")
    message: Mapped[str] = mapped_column(Text)
    file_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    job: Mapped[ImportJob] = relationship(back_populates="events")


class ImportJobFile(Base):
    """Persistent per-file state for resumable directory synchronization."""

    __tablename__ = "import_job_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("import_jobs.id", ondelete="CASCADE"),
        index=True,
    )
    file_path: Mapped[str] = mapped_column(String(2048), index=True)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    phase: Mapped[str] = mapped_column(String(128), default="等待处理")
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    modified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    job: Mapped[ImportJob] = relationship(back_populates="files")


def get_engine(settings: Settings | None = None):
    """Create a SQLite engine for the configured database."""
    runtime_settings = settings or get_settings()
    runtime_settings.ensure_directories()
    return create_engine(f"sqlite:///{runtime_settings.sqlite_path}", future=True)


def init_database(settings: Settings | None = None) -> None:
    """Create all database tables if they do not exist."""
    engine = get_engine(settings)
    Base.metadata.create_all(engine)
    _ensure_sqlite_columns(engine)


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    """Return a SQLAlchemy session factory."""
    engine = get_engine(settings)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    """Provide a transactional session scope."""
    factory = get_session_factory(settings)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _ensure_sqlite_columns(engine) -> None:
    """Add newly introduced SQLite columns for existing local databases."""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if "import_jobs" not in table_names:
        return
    statements = []
    import_job_columns = {column["name"] for column in inspector.get_columns("import_jobs")}
    if "deleted" not in import_job_columns:
        statements.append("ALTER TABLE import_jobs ADD COLUMN deleted INTEGER DEFAULT 0")
    if "documents" in table_names:
        document_columns = {column["name"] for column in inspector.get_columns("documents")}
        if "index_profile" not in document_columns:
            statements.append("ALTER TABLE documents ADD COLUMN index_profile TEXT")
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
