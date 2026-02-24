from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from .librarian import LibrarianAgent
from .strategist import StrategistAgent
from .developer import DeveloperAgent


class ResearchState(TypedDict):
    pdf_path: str
    extracted_metadata: str
    suggested_strategies: List[Dict]
    selected_strategy_spec: Dict
    errors: List[str]


def build_supervisor(vec_engine):
    """Build and return the compiled supervisor workflow.

    Accepts a VectorEngine instance so that this module doesn't need
    to import from the ``tools`` package directly.
    """
    librarian = LibrarianAgent(vec_engine)
    strategist = StrategistAgent()
    developer = DeveloperAgent()

    def librarian_node(state: ResearchState):
        print("--- Librarian is searching the paper ---")
        executor = librarian.get_executor()
        query = "Extract the core alpha factors, PCA methodology, and ETF arbitrage logic from the paper."
        result = executor.invoke({"messages": [HumanMessage(content=query)]})
        final_answer = result["messages"][-1].content
        return {"extracted_metadata": final_answer}

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

    return workflow.compile()