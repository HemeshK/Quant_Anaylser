import os
import json
import pandas as pd
from agents.supervisor import supervisor_workflow
from tools.vector_engine import VectorEngine
from tools.data_fetcher import DataLoader
from tools.backtester import PCABacktester

def run_research(pdf_path):
    # --- PHASE 1: AI RESEARCH ---
    # 1. Ingest Data (Local Vectorization)
    engine = VectorEngine()
    print(f"--- Ingesting {pdf_path} ---")
    engine.ingest_pdf(pdf_path)
    
    # 2. Run Agentic Workflow (Groq Llama 3.3)
    print("--- Starting Agentic Workflow ---")
    initial_state = {
        "pdf_path": pdf_path,
        "extracted_metadata": "",
        "suggested_strategies": [],
        "selected_strategy_spec": {},
        "errors": []
    }
    
    final_output = supervisor_workflow.invoke(initial_state)
    spec = final_output.get("selected_strategy_spec", {})
    
    print("\n" + "="*50)
    print("FINAL STRATEGY SPECIFICATION:")
    print("="*50)
    print(json.dumps(spec, indent=2))

    # --- PHASE 2: DATA & MATH VALIDATION ---
    print("\n" + "="*50)
    print("RUNNING BACKTESTER VALIDATION")
    print("="*50)
    
    # Initialize our new tools
    loader = DataLoader()
    
    # We'll use a standard tech-sector sample for the PCA math
    # In a real run, the 'spec' could dictate which tickers to fetch
    sample_tickers = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "TSLA", "AVGO", "ORCL"]
    
    try:
        # 1. Fetch Data
        prices = loader.get_market_data(sample_tickers, start="2024-01-01", end="2026-02-01")
        returns = prices.pct_change().dropna()

        # 2. Run PCA Analysis (matching the 15 components identified in the paper)
        # Note: We use 5 components here because our sample universe is small
        tester = PCABacktester(n_components=5)
        residuals = tester.calculate_residuals(returns)
        z_scores = tester.compute_signals(residuals)

        print("\n--- Recent Statistical Arbitrage Z-Scores ---")
        print(z_scores.tail())
        
        # Identify "Cheap" (Long) or "Expensive" (Short) signals
        latest_signals = z_scores.iloc[-1]
        print("\n--- Trading Signals ---")
        for ticker, z in latest_signals.items():
            if z > 2:
                print(f"SELL {ticker}: Z-Score is {z:.2f} (Overbought)")
            elif z < -2:
                print(f"BUY {ticker}: Z-Score is {z:.2f} (Oversold)")
            else:
                pass # Fairly valued
                
    except Exception as e:
        print(f"Backtest engine encountered an error: {e}")

if __name__ == "__main__":
    file_to_analyze = "data/stat_arb.pdf"
    if os.path.exists(file_to_analyze):
        run_research(file_to_analyze)
    else:
        print(f"File not found: {file_to_analyze}")