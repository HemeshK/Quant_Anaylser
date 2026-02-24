import os
import json
import pandas as pd
from agents.supervisor import build_supervisor
from tools.vector_engine import VectorEngine
from tools.data_fetcher import DataLoader
from tools.backtester import PCABacktester

# 1. OPTIONAL: Semantic Caching for speed
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

def run_research(pdf_path):
    # --- PHASE 1: AI RESEARCH ---
    # 1. Ingest Data (Local Vectorization)
    engine = VectorEngine()
    print(f"--- Ingesting {pdf_path} ---")
    engine.ingest_pdf(pdf_path)
    
    # 2. Build and Run Agentic Workflow
    print("--- Starting Agentic Workflow ---")
    supervisor_workflow = build_supervisor(engine)
    initial_state = {
        "pdf_path": pdf_path,
        "extracted_metadata": "",
        "suggested_strategies": [],
        "selected_strategy_spec": {},
        "errors": []
    }
    
    # supervisor_workflow handles the logic between agents
    final_output = supervisor_workflow.invoke(initial_state)
    spec = final_output.get("selected_strategy_spec", {})
    
    print("\n" + "="*50)
    print("FINAL STRATEGY SPECIFICATION:")
    print("="*50)
    print(json.dumps(spec, indent=2))

    # --- PHASE 2: DATA & DYNAMIC VALIDATION ---
    print("\n" + "="*50)
    print("RUNNING DYNAMIC BACKTESTER VALIDATION")
    print("="*50)
    
    loader = DataLoader()
    
    # Universe selection can now be guided by the spec if implemented
    sample_tickers = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "TSLA", "AVGO", "ORCL"]
    
    try:
        # 1. Fetch Data (Extended window for better moving averages)
        prices = loader.get_market_data(sample_tickers, start="2024-01-01", end="2026-02-12")
        returns = prices.pct_change().dropna()

        # 2. DYNAMIC BACKTESTING
        # We pass the 'spec' directly so the math adapts to the paper's findings
        tester = PCABacktester(spec=spec) 
        
        print(f"--- Using {tester.n_components} PCA factors and {tester.window}-day window ---")
        
        residuals = tester.calculate_residuals(returns)
        z_scores = tester.compute_signals(residuals)

        print("\n--- Recent Statistical Arbitrage Z-Scores ---")
        print(z_scores.tail())
        
        # 3. Dynamic Thresholding
        latest_signals = z_scores.iloc[-1]
        # Use threshold from spec or default to 2.0
        threshold = spec.get("strategy", {}).get("parameters", {}).get("threshold", 2.0)
        
        print(f"\n--- Trading Signals (Threshold: {threshold}) ---")
        for ticker, z in latest_signals.items():
            if z > threshold:
                print(f"SELL {ticker}: Z-Score is {z:.2f} (Overbought)")
            elif z < -threshold:
                print(f"BUY {ticker}: Z-Score is {z:.2f} (Oversold)")
                
    except Exception as e:
        print(f"Backtest engine encountered an error: {e}")

if __name__ == "__main__":
    file_to_analyze = "data/test_paper.pdf"
    if os.path.exists(file_to_analyze):
        run_research(file_to_analyze)
    else:
        print(f"File not found: {file_to_analyze}")