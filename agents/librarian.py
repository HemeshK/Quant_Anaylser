import os
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv

load_dotenv()
class LibrarianAgent:
    def __init__(self, vector_engine):
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Create a tool the agent can use to "search" the PDF
        retriever = vector_engine.get_retriever()
        self.tool = create_retriever_tool(
            retriever,
            "search_quant_paper",
            "Searches the stat-arb paper for specific alpha factors and PCA logic."
        )
        self.tools = [self.tool]

    def get_executor(self):
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)