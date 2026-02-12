import os
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class LibrarianAgent:
    def __init__(self, vector_engine):
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile", # Good choice for speed/cost
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        retriever = vector_engine.get_retriever()
        self.tool = create_retriever_tool(
            retriever,
            "search_quant_paper",
            "Use this to search the PDF. Search for ONE concept at a time (e.g., just 'PCA' or just 'alpha factors')."
        )
        self.tools = [self.tool]

    def get_executor(self):
        # Custom ReAct prompt to stop the hallucination loop
        template = """You are a Quantitative Research Librarian. Your goal is to extract technical details from a PDF.
        
        CRITICAL RULES:
        1. If a search result is repetitive or unhelpful, CHANGE your search query to be more specific (e.g., search for 'formula' or 'equation').
        2. Do not search for multiple concepts at once. Break them down.
        3. You must provide a final answer that includes the PCA methodology, alpha factors, and ETF logic.

        TOOLS:
        ------
        You have access to the following tools:
        {tools}

        To use a tool, please use the following format:
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action

        When you have a response for the user, or if you cannot find more info:
        Thought: Do I need to use a tool? No
        Final Answer: [your detailed summary here]

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=10 # Prevents infinite loops
        )