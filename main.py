import os
import streamlit as st
import pickle
import time
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

#api calling git ls-files --stage
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0.7, max_tokens=500)

# Set Streamlit page layout
st.set_page_config(page_title="Research Mate", layout="wide")

#title with color
st.markdown(
    "<h1 style='text-align: center; color: #FF5733;'>Research Mate📈</h1>",
    unsafe_allow_html=True
)

st.sidebar.markdown("<h2 style='color:#3498db;'>🔗 Enter Reseach URLs</h2>", unsafe_allow_html=True)

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("🚀 Process URLs")
clear_results = st.sidebar.button("🗑️ Clear Results")
file_path = "faiss_store_groq.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    if any(urls):
        main_placeholder.markdown("<h3 style='color:green;'>✅ Data Loading...Started...</h3>", unsafe_allow_html=True)
        
        # Load data from URLs
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        # Split the data into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.markdown("<h3 style='color:green;'>✅ Text Splitter...Started...</h3>", unsafe_allow_html=True)
        docs = text_splitter.split_documents(data)

        # Create embeddings using HuggingFace
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore_groq = FAISS.from_documents(docs, embeddings)
        main_placeholder.markdown("<h3 style='color:green;'>✅ Embedding Vector Started Building...</h3>", unsafe_allow_html=True)
        time.sleep(2)

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_groq, f)

        st.sidebar.success("✅ URLs processed successfully!")

    else:
        st.sidebar.error("⚠️ Please enter at least one URL.")

query = st.text_input("💡 Ask a Question:", placeholder="Type your question here...")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            # Use RetrievalQAWithSourcesChain with the Groq LLM
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # Display answer with a colored background
            st.markdown("<h2 style='color:#27ae60;'>📜 Answer:</h2>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='background-color:#f4f4f4; padding:15px; border-radius:10px; color:#333;'>{result['answer']}</div>",
                unsafe_allow_html=True
            )

            # Display sources if available
            sources = result.get("sources", "")
            if sources:
                st.markdown("<h3 style='color:#3498db;'>📌 Sources:</h3>", unsafe_allow_html=True)
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.markdown(f"- 🔗 {source}")

if clear_results:
    st.experimental_rerun()
