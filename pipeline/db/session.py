import os
from typing import Generator
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

def get_database_url() -> str:
    """
    Construct database connection URL supporting:
    1. Standard DATABASE_URL env var (e.g., local Docker or direct URI).
    2. Cloud SQL unix socket if DB_INSTANCE is provided (Cloud Run production).
    """
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        # Standardize postgres:// or postgresql:// to postgresql+psycopg2:// for SQLAlchemy
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif db_url.startswith("postgresql://") and not db_url.startswith("postgresql+"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return db_url

    # Check Cloud SQL connection parameters (from Cloud Run environment variables)
    db_instance = os.getenv("DB_INSTANCE")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "orbit")

    if db_instance:
        # In Cloud Run, Cloud SQL socket is mounted at /cloudsql/<INSTANCE_CONNECTION_NAME>
        socket_path = f"/cloudsql/{db_instance}"
        return f"postgresql+psycopg2://{db_user}:{db_password}@/{db_name}?host={socket_path}"

    # Default fallback for local testing
    return "postgresql+psycopg2://postgres:postgres@127.0.0.1:5433/orbit"



DATABASE_URL = get_database_url()

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=1800,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def is_db_configured() -> bool:
    """Verify if the database is reachable."""
    try:
        with engine.connect() as conn:
            return True
    except Exception:
        return False

def get_db() -> Generator:
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
