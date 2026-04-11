"""
run_stability_test.py
---------------------
Orchestrator for Yearly Sensitivity Analysis.
Runs the backtest engine across multiple start years to evaluate strategy stability.
"""

import os
import pandas as pd
from run_backtest import main as run_main

# ===========================================================================
# 1. Sensitivity Configuration
# ===========================================================================
SENSITIVITY_YEARS = list(range(2009, 2015)) # 2009, 2010, 2011, 2012, 2013, 2014
HURST_WINDOW = 200  # Target window for testing

def run_sensitivity_analysis():
    print("=" * 70)
    print("  YEARLY STABILITY TEST (WALK-FORWARD SENSITIVITY)")
    print("=" * 70)
    
    summary_results = []
    
    for year in SENSITIVITY_YEARS:
        print(f"\n>>> LAUNCHING BACKTEST FOR START YEAR: {year} <<<")
        
        # Subdirectory for results to prevent overwriting
        subdir = f"stability_test/Y{year}"
        
        # Execute the backtest for this year
        kpis = run_main(hurst_window=HURST_WINDOW, start_year=year, output_subdir=subdir)
        
        # Tag results with the year and store
        for item in kpis:
            item['Launch_Year'] = year
            summary_results.append(item)
        
        print(f"\n[DONE] Backtest for {year} complete. Results in output/{subdir}")

    # 2. Consolidated Comparison Table
    print("\n" + "=" * 130)
    print("  FINAL YEARLY STABILITY COMPARISON")
    print("=" * 130)
    print(f"{'Strategy [Launch Year]':<65} | {'Final Value':>15} | {'Trades':>8} | {'Win%':>8} | {'MaxDD':>8} | {'Sharpe':>7}")
    print("-" * 130)
    
    # Sort by year then strategy name for readability
    summary_results.sort(key=lambda x: (x['Launch_Year'], x['Strategy']))
    
    for r in summary_results:
        label = f"{r['Strategy']} [{r['Launch_Year']}]"
        print(
            f"{label:<65} | ${r['Final Value']:>14,.2f} | "
            f"{r['Trades']:>8} | {r['Win Rate (%)']:>7.1f}% | "
            f"{r['Max DD (%)']:>7.2f}% | {r['Sharpe']:>7.2f}"
        )
    print("-" * 130)
    print(f"Total results saved in the output/stability_test/ directory.")

if __name__ == "__main__":
    run_sensitivity_analysis()
