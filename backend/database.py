import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# This line loads the variables from your .env file
load_dotenv()

# This line securely gets the connection string using your specific variable name "db_url"
DATABASE_URL = os.getenv("db_url")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()