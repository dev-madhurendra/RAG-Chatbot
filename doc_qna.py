
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import chromadb
from langchain.text_splitter import CharacterTextSplitter
from chromadb.utils import embedding_functions
import openai
import os

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize OpenAI embedding function
openai_embedding_functions = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-ada-002", 
    api_key=OPENAI_API_KEY
)

# Connect to ChromaDB and create a collection with the OpenAI embedding function
chroma_client = chromadb.Client()
chroma_client.delete_collection("chatbot_collection")
collection = chroma_client.create_collection(
    name="chatbot_collection", 
    embedding_function=openai_embedding_functions
)


# Function to extract text from PDF files
def extract_text_from_pdf(pdf_docs):
    """Extract text from PDF files and concatenate it."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(raw_text):
    """Split raw text into chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800, 
        chunk_overlap=200, 
        length_function=len
    )
    return text_splitter.split_text(raw_text)

# Function to add text chunks to ChromaDB collection
def add_text_chunks_to_collection(text_chunks):
    """Add text chunks to ChromaDB collection."""
    documents = text_chunks
    text_embeddings = openai_embedding_functions(documents)
    ids = ["item_id" + str(i) for i in range(len(text_chunks))]  # Generate unique IDs for each chunk
    doc_metadata = [
        {
            "chunk_id": id,
            "source": "PDF",
            "document_name": "sample.pdf"  # Specify the name of the PDF document
        }
        for id in ids
    ]
    
    collection.add(
        embeddings=text_embeddings, 
        documents=documents, 
        ids=ids,
        metadatas=doc_metadata
    )
    return chroma_client

# Function to get completions from OpenAI chat model
def get_chat_completion(model, system_msg, query):
    """Get completions from OpenAI chat model."""
    return openai.Client(api_key=OPENAI_API_KEY).chat.completions.create(
        model=model,
        messages=[
            {
                'role': "system", 
                "content": system_msg
            },
            {
                "role": "user", 
                "content": query
            }
        ],
        temperature=0.0,
        max_tokens=150,
    )

# Function to retrieve query results from ChromaDB
def retrieve_query_results_from_db(user_query):
    """Retrieve query results from ChromaDB."""
    results = collection.query(query_texts=[user_query], n_results=1)
    print(">>> vector db query ",  results)
    
    context_msg = "\n".join(doc[0] for doc in results['documents'])
    print(">>> context message ", context_msg)
    return interact_with_open_ai_chat_modal(user_query, context_msg)

# Function to interact with OpenAI chatbot
def interact_with_open_ai_chat_modal(query, context_msg):
    """Interact with OpenAI chatbot."""
    system_msg = f""" You are a dedicated support agent. Please provide accurate responses based on the given context.
    Your task is to address inquiries within the provided context. Ensure that your answers align with the context of the question.
    Avoid speculation and provide responses only to the questions asked.
    If a question is unrelated to the context, respond with 'Out of Context'.
    Below is the provided context:
    {context_msg} """

    response = get_chat_completion("gpt-3.5-turbo" , system_msg, query)
    chatbot_response = response.choices[0].message.content.strip()
    return chatbot_response

# Streamlit app
def main():
    st.title("RAG Use Case")

    pdf_file = st.file_uploader("Upload PDF File", type=["pdf"])

    if pdf_file:
        st.write("PDF File Uploaded Successfully!")
        pdf_text = extract_text_from_pdf([pdf_file])
        text_chunks = split_text_into_chunks(pdf_text)
        add_text_chunks_to_collection(text_chunks)
        st.write("Text extracted and added to collection!")

    user_query = st.text_input("You:", "")
    if st.button("Send"):
        chatbot_response = retrieve_query_results_from_db(user_query)
        st.write("ChatBot:", chatbot_response)

if __name__ == "__main__":
    main()
