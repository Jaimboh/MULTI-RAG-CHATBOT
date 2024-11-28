import re
import streamlit as st
import pandas as pd
from langtrace_python_sdk import langtrace
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from Prompts.prompts import prompt_template_for_question
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from config.settings import get_settings
import warnings


warnings.filterwarnings("ignore")

# Get settings
settings = get_settings()

#langtrace configuration
langtrace.init(api_key = settings.langtrace.api_key)

#Model Configuration
llm = ChatGroq(model_name=settings.groq.default_model, api_key=settings.groq.api_key)

# Database Configuration
IMAGE_DB_NAME = settings.database.image_db_name
PDF_DB_NAME = settings.database.pdf_db_name
PDF_COLLECTION_NAME = settings.database.pdf_collection_name

# Load grocery data
df = pd.read_csv('Data/csv_data/grocery_data.csv')
df = df[['description', 'categoryName', 'categoryID', 
         'price', 'nutritions','name']]

# Database connection string
def get_connection_string(db_name):
    return f"postgresql+psycopg2://{settings.database.user}:{settings.database.password}@{settings.database.host}:{settings.database.port}/{db_name}"

def search_image_database(query: str, results: int = 4):
    try:
        # Initialize OpenAI embeddings for image search
        embeddings = OpenAIEmbeddings(
            model=settings.openai.embedding_model,
            dimensions=settings.vector_store.embedding_dimensions
        )

        connection_string = get_connection_string(IMAGE_DB_NAME)

        # Create SQLAlchemy engine
        engine = create_engine(connection_string)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # Generate embedding for the query
            query_embedding = embeddings.embed_query(query)

            # Ensure the query embedding is cast to the vector type
            query_embedding_vector = f"[{','.join(map(str, query_embedding))}]"

            # Query using cosine similarity
            query = text("""
                SELECT image_path, embedding <=> :embedding AS distance
                FROM image_embeddings
                ORDER BY embedding <=> :embedding
                LIMIT :k
            """)

            results = session.execute(
                query,
                {
                    "embedding": query_embedding_vector,
                    "k": results
                }
            )

            # Extract image paths directly from results
            image_paths = [row[0] for row in results]

            if image_paths:
                st.write(f"Found {len(image_paths)} matching images")
                for path in image_paths:
                    st.write(f"Loading image from: {path}")

            else:
                st.warning("No matching images found")

            return image_paths

        finally:
            session.close()

    except Exception as e:
        st.error(f"Error querying the image database: {str(e)}")
        return None

def check_if_data_needed(user_input: str) -> str:
    prompt = """The user has asked: "{user_input}".
    Determine the appropriate response based on the question.
    - If the question is related to **Cloth image datasets**, respond with "Image".
    - If it is related to **grocery store items**, respond with "Grocery".
    - If it is related to **Finetunning LLM Models**, respond with "Finetunning"
    - else, respond with **normal**

    Output should be **one word**: "Image" or "Grocery" or "Finetunning" or "normal".
    """
    template = PromptTemplate(template=prompt, input_variables=["user_input"])
    chain = template | llm
    decision = chain.invoke({"user_input": user_input}).content.strip().lower()
    print("decision --------------------> ", decision)
    return decision

def get_data_from_csv(user_query: str) -> str:
    prompt = PromptTemplate(template=prompt_template_for_question, input_variables=["user_query"])
    chain = prompt | llm
    output = chain.invoke({"user_query": user_query})

    pattern = r"Python Code: ```(.*?)```"
    matches = re.findall(pattern, output.content, re.DOTALL)

    if matches:
        try:
            result = eval(matches[0])
            return result.to_json(orient='records')
        except Exception:
            return "Error in retrieving data."
    return "No matching data found."

def respond_to_user(user_input: str):
    decision = check_if_data_needed(user_input)

    if decision == "grocery":
        retrieved_data = get_data_from_csv(user_input)
        st.write(f"Retrieved Grocery Data:\n{retrieved_data}")

    elif decision == "image":
        image_paths = search_image_database(user_input, results=4)
        if image_paths:
            st.write("Image Search Results:")
            for path in image_paths:
                st.image(path)

    elif decision == "finetunning":
        # Initialize OpenAI embeddings for PDF search
        embeddings = OpenAIEmbeddings(
            model=settings.openai.embedding_model,
            dimensions=settings.vector_store.embedding_dimensions
        )

        # Connect to PGVector for PDF documents
        pdf_vector_store = PGVector(
            connection_string=get_connection_string(PDF_DB_NAME),
            embedding_function=embeddings,
            collection_name=PDF_COLLECTION_NAME
        )

        vectorstore_retriever = pdf_vector_store.as_retriever(
            search_kwargs={"k": 1}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore_retriever,
            return_source_documents=True
        )

        question = "You are a helpful AI Assistant. Your job is to generate output based on the query."
        query = question + " Requirement: " + user_input
        llm_response = qa_chain(query)
        st.write(llm_response["result"])

    else:
        prompt = PromptTemplate(
            template="You are a helpful Assistant. Respond to the user's query: '{user_input}'. Don't add products details from your side. Just Ask the user How can you help with greetings.",
            input_variables=["user_input"]
        )
        chain = prompt | llm
        output = chain.invoke({"user_input": user_input})
        response = output.content
        st.write(response)

# Streamlit UI
st.title("AI Assistant: One chatBot for MultiDataBases")

user_query = st.text_input("Enter your query:")
if st.button("Submit"):
    if user_query:
        respond_to_user(user_query)
    else:
        st.warning("Please enter a query.")
