import os
import streamlit as st 
import pickle
import time
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv  # Only keep if you are using environment variables

load_dotenv()  # Only call if you're using environment variables

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []  # Have to create a main so streamlit can work successfully
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()  # To show the progress bar below "News Research Tool"
llm = OpenAI(temperature=0.5, max_tokens=200)

if process_url_clicked:
    # Load the data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    # Split the data
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    separators=["\n\n", "\n", ".", ","]  # Use `separators` instead of `separator`
)

    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    # Create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
 
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
    
    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)


query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorestore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever= vectorestore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)