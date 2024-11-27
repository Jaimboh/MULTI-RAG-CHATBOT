import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pathlib import Path
from config.settings import get_settings
import PyPDF2
import os

# Get settings
settings = get_settings()

# Database configuration
BASE_CONNECTION_STRING = (
    f"user={settings.database.user} password={settings.database.password} "
    f"host={settings.database.host} port={settings.database.port}"
)
NEW_DB_NAME = settings.database.pdf_db_name
COLLECTION_NAME = settings.database.pdf_collection_name



def create_database():
    # Connect to PostgreSQL server without specifying a database
    conn = psycopg2.connect(BASE_CONNECTION_STRING)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (NEW_DB_NAME,))
    exists = cursor.fetchone()
    
    if not exists:
        # Create new database
        cursor.execute(f'CREATE DATABASE "{NEW_DB_NAME}"')
        
    cursor.close()
    conn.close()
    
    # Connect to the new database to create the pgvector extension
    conn = psycopg2.connect(f"{BASE_CONNECTION_STRING} dbname={NEW_DB_NAME}")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Create pgvector extension if it doesn't exist
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    cursor.close()
    conn.close()

def load_pdf_text_pypdf2(file_path):
    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def process_documents(directory):
    documents = []
    for pdf_file in Path(directory).glob("*.pdf"):
        text = load_pdf_text_pypdf2(pdf_file)
        documents.append(Document(
            page_content=text,
            metadata={"source": str(pdf_file)}
        ))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    return text_splitter.split_documents(documents)


# Create database and set up pgvector
create_database()

# Set up connection string for PGVector
CONNECTION_STRING = f"postgresql+psycopg2://{settings.database.user}:{settings.database.password}@{settings.database.host}:{settings.database.port}/{NEW_DB_NAME}"


# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model=settings.openai.embedding_model,
    dimensions=settings.vector_store.embedding_dimensions
)

# Initialize PGVector
vector_store = PGVector.from_documents(
    embedding=embeddings,
    documents=process_documents("Data/pdf_data"),
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

print(f"PDF documents embedded and saved to PostgreSQL database '{NEW_DB_NAME}'")

