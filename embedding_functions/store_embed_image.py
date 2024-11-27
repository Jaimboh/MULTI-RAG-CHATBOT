import os
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import numpy as np
from PIL import Image
import io
import base64
from langchain_openai import OpenAIEmbeddings
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from config.settings import get_settings

# Get settings
settings = get_settings()


# Database configuration
BASE_CONNECTION_STRING = (
    f"user={settings.database.user} password={settings.database.password} "
    f"host={settings.database.host} port={settings.database.port}"
)
NEW_DB_NAME = settings.database.image_db_name


def create_database():
    """Create the database if it doesn't exist"""
    conn = psycopg2.connect(BASE_CONNECTION_STRING)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (NEW_DB_NAME,))
    exists = cursor.fetchone()
    
    if not exists:
        cursor.execute(f'CREATE DATABASE "{NEW_DB_NAME}"')
    
    cursor.close()
    conn.close()
    
    # Connect to the new database to create the pgvector extension
    conn = psycopg2.connect(f"{BASE_CONNECTION_STRING} dbname={NEW_DB_NAME}")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cursor.close()
    conn.close()

# Create database and tables
create_database()

# SQLAlchemy setup
DATABASE_URL = f"postgresql+psycopg2://{settings.database.user}:{settings.database.password}@{settings.database.host}:{settings.database.port}/{NEW_DB_NAME}"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

class ImageEmbedding(Base):
    __tablename__ = settings.vector_store.table_name

    id = Column(Integer, primary_key=True)
    image_path = Column(String)
    embedding = Column(Vector(settings.vector_store.embedding_dimensions))

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def image_to_base64(image_path):
    """Convert image to base64 string"""
    with Image.open(image_path) as image:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image if too large (OpenAI has size limits)
        max_size = (1024, 1024)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def process_image(image_path, embeddings_model):
    """Process a single image and return its embedding using OpenAI"""
    try:
        # Convert image to base64
        image_base64 = image_to_base64(image_path)
        
        # Create a text description of the image that OpenAI can embed
        image_text = image_base64
        
        # Get embedding using OpenAI
        embedding = embeddings_model.embed_query(image_text)
        return np.array(embedding)
    
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def main():
    dataset_folder = 'Data/image_data'
    session = Session()
    
    # Initialize OpenAI embeddings
    embeddings_model = OpenAIEmbeddings(
        model=settings.openai.embedding_model,
        dimensions=settings.vector_store.embedding_dimensions
    )
    
    try:
        # Process all PNG images in the folder
        for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
            if filename.endswith('.png'):
                file_path = os.path.join(dataset_folder, filename)
                
                # Get image embedding
                embedding = process_image(file_path, embeddings_model)
                
                if embedding is not None:
                    # Store in database
                    image_embedding = ImageEmbedding(
                        image_path=file_path,
                        embedding=embedding.astype(float)
                    )
                    session.add(image_embedding)
                
                # Commit every 100 images
                if i > 0 and i % 100 == 0:
                    session.commit()
                    print(f"Processed {i} images...")
        
        # Final commit
        session.commit()
        print("All images processed and stored in the vector database.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        session.rollback()
    
    finally:
        session.close()

def similarity_search(query_image_path, k=5):
    """Search for similar images"""
    session = Session()
    
    # Initialize OpenAI embeddings
    embeddings_model = OpenAIEmbeddings(
        model=settings.openai.embedding_model,
        dimensions=settings.vector_store.embedding_dimensions
    )
    
    try:
        # Get query image embedding
        query_embedding = process_image(query_image_path, embeddings_model)
        
        if query_embedding is None:
            return []
        
        # Perform similarity search
        results = session.query(ImageEmbedding).order_by(
            ImageEmbedding.embedding.cosine_distance(query_embedding.astype(float))
        ).limit(k).all()
        
        return [(result.image_path, result.id) for result in results]
    
    finally:
        session.close()

if __name__ == "__main__":
    main()
