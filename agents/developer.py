import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI # Ensure this is installed
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

class DeveloperAgent:
    def __init__(self):
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

        # 3. Create Resilient LLM
        self.llm = self.primary_llm.with_fallbacks([self.fallback_llm])

    def write_spec(self, state):
        strategies = state.get("suggested_strategies", [])
        
        if isinstance(strategies, dict):
            strategies = strategies.get("strategies", strategies.get("suggested_strategies", []))
            
        if not strategies:
            return {"errors": ["No strategies found to develop."]}

        choice = strategies[0]
        
        prompt = ChatPromptTemplate.from_template(
            "System: You are an Algorithmic Developer. Output ONLY valid JSON.\n"
            "Human: Create a detailed technical specification (ticker, entry, exit, risk) for this strategy: {choice}"
        )
        
        # Chain now uses the resilient llm with built-in fallback logic
        chain = prompt | self.llm | JsonOutputParser()
        result = chain.invoke({"choice": choice})
        return {"selected_strategy_spec": result}