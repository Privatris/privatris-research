import sys
import os
# Add project root to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
from src.train import run_multiple_seeds

def run_ablation_study(steps=50, seeds=10):
    print("Starting Ablation Study: PID vs Gradient Ascent")
    print(f"Configuration: {steps} steps, {seeds} seeds per controller")
    print("-" * 60)

    # 1. Run PID (PRIVATRIS)
    print("\n>>> Running PRIVATRIS (PID Controller)...")
    pid_results = run_multiple_seeds(num_seeds=seeds, steps=steps, controller='pid')
    
    # 2. Run Gradient Ascent (Baseline)
    print("\n>>> Running Baseline (Gradient Ascent)...")
    ga_results = run_multiple_seeds(num_seeds=seeds, steps=steps, controller='gradient_ascent')

    # 3. Compare Results
    pid_svr = [r["svr_final"] for r in pid_results]
    ga_svr = [r["svr_final"] for r in ga_results]
    
    pid_utility = [r["utility_avg"] for r in pid_results]
    ga_utility = [r["utility_avg"] for r in ga_results]

    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*60)
    print(f"{'Metric':<20} | {'PRIVATRIS (PID)':<20} | {'Baseline (GA)':<20}")
    print("-" * 66)
    
    # SVR Stats
    pid_svr_mean = np.mean(pid_svr) * 100
    pid_svr_std = np.std(pid_svr) * 100
    ga_svr_mean = np.mean(ga_svr) * 100
    ga_svr_std = np.std(ga_svr) * 100
    
    print(f"{'SVR (%)':<20} | {pid_svr_mean:.2f} ± {pid_svr_std:.2f}      | {ga_svr_mean:.2f} ± {ga_svr_std:.2f}")
    
    # Utility Stats
    pid_util_mean = np.mean(pid_utility)
    pid_util_std = np.std(pid_utility)
    ga_util_mean = np.mean(ga_utility)
    ga_util_std = np.std(ga_utility)
    
    print(f"{'Utility':<20} | {pid_util_mean:.2f} ± {pid_util_std:.2f}      | {ga_util_mean:.2f} ± {ga_util_std:.2f}")
    
    # Variance Reduction Calculation
    svr_variance_reduction = (1 - (pid_svr_std**2 / (ga_svr_std**2 + 1e-10))) * 100
    print("-" * 66)
    print(f"Stability Improvement (Variance Reduction): {svr_variance_reduction:.1f}%")
    print("="*60)

if __name__ == "__main__":
    # Default to 50 steps for quick verification, 10 seeds as requested
    run_ablation_study(steps=50, seeds=10)
