from langgraph.graph import StateGraph, END
from .librarian import LibrarianAgent
from .strategist import StrategistAgent
from .developer import DeveloperAgent
from tools.vector_engine import VectorEngine
from state import ResearchState

# Initialize tools and agents
vec_engine = VectorEngine()
librarian = LibrarianAgent(vec_engine)
strategist = StrategistAgent()
developer = DeveloperAgent()

def librarian_node(state: ResearchState):
    print("--- Librarian is searching the paper ---")
    executor = librarian.get_executor()
    query = "Extract the core alpha factors, PCA methodology, and ETF arbitrage logic from the paper."
    result = executor.invoke({"input": query})
    return {"extracted_metadata": result["output"]}

def strategist_node(state: ResearchState):
    print("--- Strategist is generating ideas ---")
    return strategist.generate_ideas(state)

def developer_node(state: ResearchState):
    print("--- Developer is writing the technical spec ---")
    return developer.write_spec(state)

# Build the Graph
workflow = StateGraph(ResearchState)

workflow.add_node("librarian", librarian_node)
workflow.add_node("strategist", strategist_node)
workflow.add_node("developer", developer_node)

workflow.set_entry_point("librarian")
workflow.add_edge("librarian", "strategist")
workflow.add_edge("strategist", "developer")
workflow.add_edge("developer", END)

supervisor_workflow = workflow.compile()