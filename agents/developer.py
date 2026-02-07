import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
class DeveloperAgent:
    def __init__(self):
        # Pass the key directly here
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    def write_spec(self, state):
        # Handle different possible JSON structures from the Strategist
        strategies = state.get("suggested_strategies", [])
        
        # If the Strategist wrapped it in a dict, extract the list
        if isinstance(strategies, dict):
            # Look for common keys like 'strategies' or 'suggested_strategies'
            strategies = strategies.get("strategies", strategies.get("suggested_strategies", []))
            
        if not strategies:
            return {"errors": ["No strategies found to develop."]}

        choice = strategies[0] # Now it's safe to pick the first one
        
        prompt = ChatPromptTemplate.from_template(
            "System: You are an Algorithmic Developer. Output ONLY valid JSON.\n"
            "Human: Create a detailed technical specification (ticker, entry, exit, risk) for this strategy: {choice}"
        )
        chain = prompt | self.llm | JsonOutputParser()
        result = chain.invoke({"choice": choice})
        return {"selected_strategy_spec": result}