import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorEngine:
    def __init__(self, collection_name="quant_papers"):
        self.persist_directory = "./chroma_db"
        self.collection_name = collection_name
        # The embedding model remains local
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def ingest_pdf(self, file_path):
        """Only ingests if the persistent directory is empty."""
        # Check if the database folder already exists and has files
        if os.path.exists(self.persist_directory) and len(os.listdir(self.persist_directory)) > 0:
            print(f"--- Existing index found in {self.persist_directory}. Skipping ingestion. ---")
            return "Using existing index."

        if not os.path.exists(file_path):
            return f"Error: {file_path} not found."
            
        print(f"--- No index found. Ingesting {file_path} ---")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        
        # This saves the data to the persist_directory
        Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        print(f"--- Successfully indexed {len(splits)} chunks locally ---")
        return f"Successfully ingested {len(splits)} chunks."

    def get_retriever(self):
        """Connects to the persistent database on disk."""
        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        return vector_store.as_retriever(search_kwargs={"k": 5})