import warnings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

warnings.filterwarnings("ignore", category=DeprecationWarning)

INDEX_DIR = "faiss_index"


def fetch_answer_from_llm(query: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=False
    )

    return qa_chain.run(query)


def chat():
    print("📢 Chatbot ready! Type exit to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        answer = fetch_answer_from_llm(query)
        print(f"\n🤖 Bot: {answer}\n")


if __name__ == "__main__":
    chat()
