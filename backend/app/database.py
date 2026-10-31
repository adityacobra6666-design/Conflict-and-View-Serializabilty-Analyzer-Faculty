import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if DATABASE_URL:
    # Render PostgreSQL specifies postgres:// which SQLAlchemy 2.0 requires as postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )
else:
    custom_db_path = os.getenv("SQLITE_PATH", "").strip()
    if custom_db_path:
        db_path = os.path.abspath(custom_db_path)
    else:
        sqlite_dir = os.getenv("SQLITE_DIR", "").strip()
        if sqlite_dir:
            db_path = os.path.abspath(os.path.join(sqlite_dir, "sqlite.db"))
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "sqlite.db")

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.abspath(db_path)}"

    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

