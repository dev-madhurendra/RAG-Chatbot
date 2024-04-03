
import streamlit as st
from PyPDF2 import PdfReader
import chromadb
from langchain.text_splitter import CharacterTextSplitter
from chromadb.utils import embedding_functions
from config import OPENAI_API_KEY
from get_chat_completion import get_chat_completion
import openai

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

# Streamlit app
def main():
    st.title("Chatbot")

    pdf_file = st.file_uploader("Upload PDF File", type=["pdf"])

    if pdf_file:
        st.write("PDF File Uploaded Successfully!")
        
        # Extract text from PDF files
        pdf_text = ""
        pdf_docs = [pdf_file]
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
                
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800, 
            chunk_overlap=200, 
            length_function=len
        )     
        text_chunks = text_splitter.split_text(pdf_text)
            
        # Add text chunks to vector db collection    
        documents = text_chunks
        text_embeddings = openai_embedding_functions(documents)
        ids = ["id " + str(i) for i in range(len(text_chunks))]  # Generate unique IDs for each chunk
        pdf_metadata = [
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
            metadatas=pdf_metadata
        )
        
        st.write("Text extracted and added to collection!")

    user_query = st.text_input("You:", "")
    if st.button("Send"):
        
        # Retrieve query from vector database
        results = collection.query(query_texts=[user_query], n_results=1)
        context_msg = "\n".join(doc[0] for doc in results['documents'])
        
        system_msg = f""" You are a dedicated support agent. Please provide accurate responses based on the given context.
            Your task is to address inquiries within the provided context. Ensure that your answers align with the context of the question.
            Avoid speculation and provide responses only to the questions asked.
            If a question is unrelated to the context, respond with 'Out of Context'.
            Below is the provided context: {context_msg} """

        response = get_chat_completion("gpt-3.5-turbo" , system_msg, user_query)
        chatbot_response = response.choices[0].message.content.strip()
        
        st.write("ChatBot:", chatbot_response)

if __name__ == "__main__":
    main()
