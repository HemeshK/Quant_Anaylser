import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI  # New import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

class StrategistAgent:
    def __init__(self):
        # 1. Primary Model: Groq Llama 3.3
        self.primary_llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # 2. Fallback Model: Gemini 3 Flash
        # Ensure you've installed: pip install langchain-google-genai
        self.fallback_llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )

        # 3. Create a resilient LLM chain
        # This will automatically try Gemini if Groq returns a RateLimitError
        self.llm = self.primary_llm.with_fallbacks([self.fallback_llm])

    def generate_ideas(self, state):
        metadata = state.get("extracted_metadata", "")
        prompt = ChatPromptTemplate.from_template(
            "System: You are a Senior Quant. Output ONLY valid JSON.\n"
            "Human: Create 3 statistical arbitrage strategies based on: {metadata}"
        )
        
        # The 'chain' now uses the resilient LLM
        chain = prompt | self.llm | JsonOutputParser()
        result = chain.invoke({"metadata": metadata})
        return {"suggested_strategies": result}