import warnings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

warnings.filterwarnings("ignore", category=DeprecationWarning)

PDF_DIR = "pdf_documents"
INDEX_DIR = "faiss_index"


def load_docs(directory):
    loader = DirectoryLoader(
        path=directory,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def insert_data():
    print("📄 Loading PDFs...")
    documents = load_docs(PDF_DIR)

    if not documents:
        raise ValueError("No valid PDF documents found")

    print(f"✅ Loaded {len(documents)} pages")

    print("✂️ Splitting into chunks...")
    docs = split_docs(documents)
    print(f"✅ Created {len(docs)} chunks")

    print("🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("📦 Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_DIR)

    print("🎉 Vector store created and saved locally!")


if __name__ == "__main__":
    insert_data()
