# ResearchMate
![Image](https://github.com/user-attachments/assets/01cae259-eaaa-4851-a6f7-02e1026e1917)
**Research Mate** is an **end-to-end generative AI** chatbot built using **Streamlit**, **LangChain**, **OpenAI**, and other modern AI techniques. It enables users to input research URLs, extracts valuable insights, processes the data, and provides answers to user queries, offering a seamless experience for researchers, students, and professionals.

The app utilizes embeddings and document retrieval techniques, along with **OpenAI’s language models**, to deliver meaningful responses based on the content extracted from research articles, web pages, or any other content provided through URLs.

## **Features** 🎉

- **URL Processing**: Input multiple research URLs, and the app will process the content for analysis.
- **Text Splitting**: Breaks down large documents into manageable chunks to enhance data processing and retrieval.
- **Embeddings**: Converts text chunks into vector embeddings using **HuggingFace** `sentence-transformers/all-MiniLM-L6-v2` model for efficient retrieval.
- **Query Answering**: Users can ask context-specific questions, and the app generates answers powered by **OpenAI’s language models**.
- **Source Display**: The app displays the sources from which the data was retrieved, ensuring full transparency.
- **Streamlit Interface**: Simple, intuitive, and interactive UI for seamless user experience.

## **Tech Stack** 🛠️

- **Streamlit**: For building interactive web applications.
- **LangChain**: A framework for creating language model-powered applications for advanced document analysis and retrieval.
- **HuggingFace Transformers**: Used for generating embeddings and processing textual data.
- **OpenAI**: Powers the question-answering feature by leveraging GPT models for content generation and contextual answering.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors, enabling fast retrieval of relevant content.
- **dotenv**: Securely manages environment variables, including API keys.

## **How It Works** 🧠

### 1. **Input Research URLs**:
   The sidebar allows users to input up to 3 URLs for the app to process. The app fetches and extracts the content from these URLs.

### 2. **Data Processing**:
   After fetching the URLs, the app uses LangChain’s `UnstructuredURLLoader` to load data from the provided links and splits it into chunks using `RecursiveCharacterTextSplitter`.

### 3. **Embeddings & Vector Store**:
   The text chunks are converted into vector embeddings using the **HuggingFace model** (`sentence-transformers/all-MiniLM-L6-v2`). These embeddings are stored in a **FAISS vector store**, making it easy to quickly retrieve relevant data during the querying phase.

### 4. **Ask Questions**:
   Once the data is processed, users can ask any question related to the research content. **OpenAI’s language models** analyze the query and retrieve the most relevant answers from the stored data.

### 5. **Display Results**:
   The app displays the answers along with relevant sources, offering complete transparency and allowing users to verify the data.

## **OpenAI Integration** 🤖

Research Mate integrates **OpenAI’s GPT models** to enhance the chatbot’s capabilities for **contextual question-answering**. OpenAI’s models are used to generate human-like responses by considering both the query and the data embedded from the research URLs.

### **How OpenAI is Used**:
- **Language Model**: OpenAI’s GPT model generates answers based on the embeddings created from the research URLs.
- **Retrieval & Answering**: OpenAI utilizes the document embeddings to provide accurate and relevant answers in response to user queries.

## **Components of the Application** 🧩

### **Streamlit UI**:
   - **Sidebar**: Allows users to input research URLs and trigger data processing.
   - **Main Section**: Displays results, answers, and relevant sources from the retrieved content.

### **LangChain Components**:
   - **UnstructuredURLLoader**: Loads content from the provided URLs.
   - **RecursiveCharacterTextSplitter**: Splits documents into smaller chunks for more efficient processing.
   - **FAISS**: Stores and searches for relevant embeddings efficiently.
   - **HuggingFaceEmbeddings**: Embeds documents into vector space for efficient retrieval and processing.
   - **RetrievalQAWithSourcesChain**: Uses the stored embeddings to answer user queries with accuracy, leveraging **OpenAI’s GPT model**.

### **OpenAI**:
   - **Question Answering**: OpenAI’s GPT model is used to answer user queries based on the processed and retrieved content.
   - **Enhanced Responses**: OpenAI’s model generates insightful, context-aware answers that make research more accessible and easier to understand.

## **Usage Example** 💡

Once the app is running:

1. Input one or more research URLs in the sidebar.
2. Click "🚀 Process URLs" to start the data extraction process.
3. Ask a question about the research content in the text input box.
4. The system will generate and display answers based on the URL content, with **OpenAI**’s language models providing refined and context-specific responses.




