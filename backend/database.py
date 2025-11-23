import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Use standard name DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is missing! Add it to your .env file.\n"
        "Example:\nDATABASE_URL=postgresql://user:pass@host:5432/dbname"
    )

# Engine with safe defaults for production
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,      # prevents stale connection crashes
    pool_recycle=1800,       # reset connections every 30 min
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Base model class
Base = declarative_base()
