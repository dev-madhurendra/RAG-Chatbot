# Import necessary libraries
import os
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
import cassio

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Get environment variables
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.environ.get("ASTRA_DB_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Read PDF file
pdf_reader = PdfReader('./data/sample.pdf')

# Initialize a variable to store raw text extracted from the PDF
raw_text = ''

# Extract text from each page of the PDF
for page in pdf_reader.pages:
    content = page.extract_text()
    if content:
        raw_text += content

# Initialize RecursiveCharacterTextSplitter for chunking text
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n', '\n', '.', ' '],
    chunk_size=800,
    chunk_overlap=200
)

# Split raw text into chunks using text splitter
chunks = text_splitter.split_text(raw_text)

# Initialize CassIO with Astra DB application token and database ID
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Initialize OpenAI language model and embeddings
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings()

# Initialize Cassandra vector store with embeddings, table name, session, and keyspace
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="pdf_vectors",
    session=None,
    keyspace="default_keyspace"
)

# Add text chunks to the Cassandra vector store
astra_vector_store.add_texts(chunks)

# Create a vector store index wrapper for the Cassandra vector store
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Initialize a variable to track whether it's the first question
is_first_question = True

# Main loop to interactively ask questions and retrieve answers
while True:
    # Prompt for the first question or subsequent questions
    prompt = "\n Enter your question (or type 'quit' to exit): " if is_first_question else "\n What's your next question (or type 'quit' to exit): "
    query_text = input(prompt).strip()

    # Check if the user wants to quit
    if query_text.lower() == "quit":
        break

    # Skip if the user enters an empty query
    if not query_text:
        continue

    # Set is_first_question to False after the first iteration
    is_first_question = False

    print("\n QUESTION: \"%s\"" % query_text)

    # Query the vector store index with the question and the OpenAI language model
    answer = astra_vector_index.query(question=query_text, llm=llm).strip()
    print("\n ANSWER: \"%s\"" % answer)

    # Print the first document by relevance
    print("FIRST DOCUMENT BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        print("    [%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))
