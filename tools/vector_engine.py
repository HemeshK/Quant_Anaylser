import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorEngine:
    def __init__(self, collection_name="quant_papers"):
        # This model runs locally on your Mac hardware
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.persist_directory = "./chroma_db"
        self.collection_name = collection_name

    def ingest_pdf(self, file_path):
        if not os.path.exists(file_path):
            return f"Error: {file_path} not found."
            
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        
        print(f"--- Indexing {len(splits)} chunks locally ---")
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        return f"Successfully ingested {len(splits)} chunks."

    def get_retriever(self):
        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        return vector_store.as_retriever(search_kwargs={"k": 5})