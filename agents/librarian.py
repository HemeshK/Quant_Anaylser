import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
load_dotenv()

LIBRARIAN_SYSTEM_PROMPT = """You are a Quantitative Research Librarian. Your goal is to extract technical details from a PDF.

CRITICAL RULES:
1. If a search result is repetitive or unhelpful, CHANGE your search query to be more specific (e.g., search for 'formula' or 'equation').
2. Do not search for multiple concepts at once. Break them down.
3. You must provide a final answer that includes the PCA methodology, alpha factors, and ETF logic.
4. If you get the same Figure or Section twice, use a different search term.
5. Search for the 'Introduction' or 'Conclusion' first to get the high-level logic."""


class LibrarianAgent:
    def __init__(self, vector_engine):
        # 1. Primary Model: Groq Llama 3.3
        self.primary_llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # 2. Fallback Model: Gemini 3 Flash
        self.fallback_llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )

        # 3. Create a resilient LLM
        # This wrapper automatically handles switching if the primary fails
        self.llm = self.primary_llm.with_fallbacks([self.fallback_llm])

        retriever = vector_engine.get_retriever()
        self.tool = create_retriever_tool(
            retriever,
            "search_quant_paper",
            "Use this to search the PDF. Search for ONE concept at a time (e.g., just 'PCA' or just 'alpha factors')."
        )
        self.tools = [self.tool]

    def get_executor(self):
        """Return a compiled agent graph (CompiledStateGraph).

        The graph accepts ``{"messages": [HumanMessage(...)]}`` as input and
        returns a dict with a ``"messages"`` key containing the conversation.
        """
        # create_agent returns a CompiledStateGraph directly.
        # system_prompt must be a plain string (or SystemMessage), not a
        # PromptTemplate.
        agent_graph = create_react_agent(
            self.llm,
            tools=self.tools,
            prompt=LIBRARIAN_SYSTEM_PROMPT,
        )

        return agent_graph