import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import os
import requests


def download_model(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)



# Download the model 
model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q4_0.bin"
local_model= "models/llama-2-7b-chat.ggmlv3.q4_0.bin"
download_model(model_url, local_model)


# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"

def main():
    st.title("CSV Chatbot with LangChain")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load CSV file
        with st.spinner("Loading CSV file..."):
            loader = CSVLoader(file_path=uploaded_file, encoding="utf-8", csv_args={'delimiter': ','})
            data = loader.load()
        
        # Split the text into Chunks
        with st.spinner("Splitting text into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
            text_chunks = text_splitter.split_documents(data)
        
        # Download Sentence Transformers Embedding From Hugging Face
        with st.spinner("Generating embeddings..."):
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        # Convert the text Chunks into embeddings and save the embeddings into FAISS Knowledge Base
        with st.spinner("Creating vector store..."):
            docsearch = FAISS.from_documents(text_chunks, embeddings)
            if not os.path.exists(DB_FAISS_PATH):
                os.makedirs(DB_FAISS_PATH)
            docsearch.save_local(DB_FAISS_PATH)
        
        # Load LLM
        llm = CTransformers(
            model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
            model_type="llama",
            max_new_tokens=512,
            temperature=0.1
        )
        
        # Initialize QA chain
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

        st.success("CSV file processed successfully! You can now ask questions.")

        chat_history = []

        # Chatbot interface
        while True:
            query = st.text_input("Enter your query (or type 'exit' to quit):")

            if query.lower() == 'exit':
                st.write('Exiting')
                break
            
            if query.strip() == '':
                continue

            with st.spinner("Generating response..."):
                result = qa.invoke({"question": query, "chat_history": chat_history})
                st.write("Response: ", result['answer'])
                chat_history.append((query, result['answer']))

if __name__ == "__main__":
    main()
