from typing import TypedDict, List, Dict

class ResearchState(TypedDict):
    pdf_path: str
    extracted_metadata: str
    suggested_strategies: List[Dict]
    selected_strategy_spec: Dict
    errors: List[str]