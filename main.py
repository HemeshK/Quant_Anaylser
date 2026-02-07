from agents.supervisor import supervisor_workflow
from tools.vector_engine import VectorEngine
import os

def run_research(pdf_path):
    # 1. Ingest Data
    engine = VectorEngine()
    print(f"--- Ingesting {pdf_path} ---")
    engine.ingest_pdf(pdf_path)
    
    # 2. Run Agentic Workflow
    print("--- Starting Agentic Workflow ---")
    initial_state = {
        "pdf_path": pdf_path,
        "extracted_metadata": "",
        "suggested_strategies": [],
        "selected_strategy_spec": {},
        "errors": []
    }
    
    final_output = supervisor_workflow.invoke(initial_state)
    
    print("\n" + "="*50)
    print("FINAL STRATEGY SPECIFICATION:")
    print("="*50)
    import json
    print(json.dumps(final_output["selected_strategy_spec"], indent=2))

if __name__ == "__main__":
    # Ensure your data folder and file exists!
    file_to_analyze = "data/stat_arb.pdf"
    if os.path.exists(file_to_analyze):
        run_research(file_to_analyze)
    else:
        print(f"File not found: {file_to_analyze}")