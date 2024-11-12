import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader  # Use the PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import time

load_dotenv()

st.title("Document Q&A")

# Groq API Key and LLM setup
groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(
    temperature=0.4,
    groq_api_key=groq_api_key,
    model_name="llama-3.1-70b-versatile"
)

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

def vector_embedding(uploaded_file):
    # Ensure embedding only happens once
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # If file is uploaded, process it by saving it temporarily
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name  # Save the path to the temporary file

            # Load the saved temporary file using PyPDFLoader
            st.session_state.loader = PyPDFLoader(temp_file_path)
            st.session_state.docs = st.session_state.loader.load()
            
            if not st.session_state.docs:
                st.write("No documents loaded. Please ensure valid PDFs are uploaded.")
                return
            
            # Split the documents into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
            
            if not st.session_state.final_documents:
                st.write("No documents were split. Please check document loading and splitting logic.")
                return
            
            # Create vector store using the documents and embeddings
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# Button to trigger document embedding
if uploaded_file is not None:
    st.sidebar.button("Start Embedding", on_click=vector_embedding, args=(uploaded_file,))

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding(uploaded_file)
    st.write("Vector Store DB Is Ready")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time :", time.process_time() - start)
    st.write(response['answer'])

    # With a streamlit expander for document similarity search
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
