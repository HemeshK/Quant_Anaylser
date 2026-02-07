import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
class StrategistAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def generate_ideas(self, state):
        metadata = state.get("extracted_metadata", "")
        prompt = ChatPromptTemplate.from_template(
            "System: You are a Senior Quant. Output ONLY valid JSON.\n"
            "Human: Create 3 statistical arbitrage strategies based on: {metadata}"
        )
        chain = prompt | self.llm | JsonOutputParser()
        result = chain.invoke({"metadata": metadata})
        return {"suggested_strategies": result}