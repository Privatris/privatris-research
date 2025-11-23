import numpy as np
import argparse
from src.agent import RLAgent
from src.cmdp import LagrangianPID, DualObjectivePPO
from src.memory import PrivacyConstrainedMemory
from src.red_team import RedTeamAgent
from src.data_loader import DataLoader
import wandb
import sys

def train(seed=42, verbose=True):
    """Train PRIVATRIS agent.
    
    Args:
        seed: Random seed for reproducibility
        verbose: Print progress logs
    
    Returns:
        dict: Metrics including final SVR, Utility, Drift
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Initialize WandB
    # wandb.init(project="privatris-reproducibility", config={"seed": seed})

    return args

def train(seed=42, verbose=True, steps=100, model='gpt2', dataset='beavertails', controller='pid'):
    """Train PRIVATRIS agent.
    
    Args:
        seed: Random seed for reproducibility
        verbose: Print progress logs
        steps: Total training steps
        model: Model name
        dataset: Dataset name
        controller: Controller type
    
    Returns:
        dict: Metrics including final SVR, Utility, Drift
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    TOTAL_STEPS = steps
    MODEL_NAME = model
    DATASET = dataset
    CONTROLLER = controller
    STATE_DIM = 768 if 'gpt2' in MODEL_NAME else 896
    
    SAFETY_THRESHOLD = 0.025  # 2.5% (lowered to trigger Lambda updates)
    LEARNING_RATE = 1e-5 # Lower LR for Transformer
    
    # Initialize Components
    memory = PrivacyConstrainedMemory(embedding_dim=STATE_DIM)
    agent = RLAgent(
        state_dim=STATE_DIM,
        action_dim=4,
        model_name=MODEL_NAME,
        device='cpu'
    )
    
    print(f"Using model: {MODEL_NAME} (state_dim={STATE_DIM})")
    print(f"Dataset: {DATASET}")
    print(f"Controller: {CONTROLLER}")
    red_team = RedTeamAgent()
    
    # Initialize data loader(s) based on dataset selection
    if DATASET == 'both':
        # Load both datasets
        data_loader_bt = DataLoader(dataset_name='beavertails')
        data_loader_bt.load_beavertails()
        
        data_loader_hh = DataLoader(dataset_name='hh-rlhf')
        data_loader_hh.load_hh_rlhf()
        
        data_loaders = [data_loader_bt, data_loader_hh]
        print(f"Multi-dataset mode: BeaverTails + HH-RLHF")
    else:
        # Single dataset
        data_loader = DataLoader(dataset_name=DATASET)
        if DATASET == 'beavertails':
            data_loader.load_beavertails()
        else:  # hh-rlhf
            data_loader.load_hh_rlhf()
        data_loaders = [data_loader]
        print(f"Single dataset mode: {DATASET}")
    
    # CMDP Optimization
    if CONTROLLER == 'pid':
        # PRIVATRIS: PID Controller (KP=0.5, KI=0.01, KD=0.1)
        lagrangian = LagrangianPID(cost_limit=SAFETY_THRESHOLD, kp=0.5, ki=0.01, kd=0.1)
    else:
        # Baseline: Standard Gradient Ascent (KP=0.1, KI=0, KD=0)
        # Equivalent to simple update: lambda = lambda + lr * error
        lagrangian = LagrangianPID(cost_limit=SAFETY_THRESHOLD, kp=0.1, ki=0.0, kd=0.0)
        
    ppo_wrapper = DualObjectivePPO(lagrangian)
    
    print(f"Starting PRIVATRIS Training with Real LLM ({TOTAL_STEPS} steps)...")
    
    violation_count = 0
    utility_scores = []
    initial_svr = None
    
    for step in range(TOTAL_STEPS):
        # Alternate between datasets if using multi-dataset mode
        if DATASET == 'both':
            current_loader = data_loaders[step % len(data_loaders)]
        else:
            current_loader = data_loaders[0]
        
        # Get Data Sample (Mixed Utility & Adversarial)
        batch = current_loader.get_batch(batch_size=1, p_adversarial=0.2)
        sample = batch[0]
        observation = sample["text"]
        is_adversarial = (sample["type"] == "adversarial")
        
        # Red Team generates novel attacks
        if is_adversarial and np.random.rand() < 0.5:
            observation = red_team.generate_attack()

        # Agent Action (using learned policy)
        current_penalty = lagrangian.get_penalty()
        # Updated to match AgentEvolver-style signature
        action, log_prob, is_safe, utility_score = agent.act(observation, current_penalty, is_adversarial=is_adversarial)
        
        # Environment Feedback
        if is_adversarial and not is_safe:
            safety_cost = 1.0  # Violation!
            violation_count += 1
            reward = -1.0
            red_team.update(success=True)  # Red team learns from success
        elif is_adversarial and is_safe:
            safety_cost = 0.0  # Safe refusal
            reward = 0.5
            red_team.update(success=False)
        else:
            # Utility query
            safety_cost = 0.0
            reward = 1.0
        
        utility_scores.append(utility_score)
        
        # Dual-Objective Update
        modified_reward = ppo_wrapper.compute_reward(reward, safety_cost)
        
        # Store experience in privacy-constrained memory (if safe)
        if is_safe:
            # Use internal encoder
            obs_embedding = agent._encode_observation(observation)
            memory.add(observation, obs_embedding)
        
        # Update agent policy (PPO) - AgentEvolver style with response
        agent.update(modified_reward, observation, action)
        
        # Update Lambda (PID)
        current_lambda = lagrangian.update(safety_cost)
        
        # Track initial SVR for drift calculation
        # Adjust drift start for shorter runs
        drift_start = int(TOTAL_STEPS * 0.1)
        if step == drift_start and initial_svr is None:
            initial_svr = violation_count / (step + 1)
        
        # Log metrics
        if verbose and step % 100 == 0:
            svr = violation_count / (step + 1)
            avg_utility = np.mean(utility_scores[-100:]) if utility_scores else 0
            print(f"Step {step}: SVR={svr:.4f}, Lambda={current_lambda:.4f}, Utility={avg_utility:.2f}")
            # wandb.log({"SVR": svr, "Lambda": current_lambda, "Utility": avg_utility})
    
    # Final metrics
    final_svr = violation_count / TOTAL_STEPS
    avg_utility = np.mean(utility_scores)
    drift = final_svr - (initial_svr if initial_svr else 0)
    
    if verbose:
        print(f"\nTraining Complete.")
        print(f"Final SVR: {final_svr:.4f}")
        print(f"Avg Utility: {avg_utility:.2f}")
        print(f"Safety Drift: {drift:.4f}")
    
    return {
        "initial_svr": initial_svr if initial_svr else 0,
        "final_svr": final_svr,
        "svr_final": final_svr,
        "utility": avg_utility,
        "utility_avg": avg_utility,
        "drift": drift,
        "lambda": lagrangian.get_penalty(),
        "lambda_final": lagrangian.get_penalty()
    }

def run_multiple_seeds(num_seeds=10, steps=100, model='gpt2', dataset='beavertails', controller='pid'):
    """Run training with multiple seeds and compute statistics."""
    print(f"Running PRIVATRIS with {num_seeds} random seeds (Controller: {controller})...\n")
    
    all_results = []
    for seed in range(num_seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed + 1}/{num_seeds}")
        print(f"{'='*60}")
        results = train(seed=seed, verbose=(seed == 0), steps=steps, model=model, dataset=dataset, controller=controller)  # Only verbose for first seed
        all_results.append(results)
    
    # Compute statistics
    svr_values = [r["svr_final"] for r in all_results]
    utility_values = [r["utility_avg"] for r in all_results]
    drift_values = [r["drift"] for r in all_results]
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({controller.upper()}) - Mean ± 95% CI")
    print(f"{'='*60}")
    print(f"SVR @ {steps} steps: {np.mean(svr_values)*100:.2f}% ± {1.96*np.std(svr_values)*100:.2f}%")
    print(f"Utility Score:   {np.mean(utility_values):.2f} ± {1.96*np.std(utility_values):.2f}")
    print(f"Drift Magnitude: +{np.mean(drift_values)*100:.2f}%")
    print(f"{'='*60}")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'Qwen/Qwen2.5-0.5B-Instruct'])
    parser.add_argument('--dataset', type=str, default='beavertails', 
                       choices=['beavertails', 'hh-rlhf', 'both'],
                       help='Dataset: beavertails, hh-rlhf, or both')
    parser.add_argument('--controller', type=str, default='pid',
                       choices=['pid', 'gradient_ascent'],
                       help='Lagrangian controller type: pid (PRIVATRIS) or gradient_ascent (Baseline)')
    parser.add_argument('--multi-seed', action='store_true', help='Run with multiple seeds')
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds for multi-seed run')
    
    args = parser.parse_args()

    if args.multi_seed:
        run_multiple_seeds(num_seeds=args.seeds, steps=args.steps, model=args.model, dataset=args.dataset, controller=args.controller)
    else:
        train(seed=42, verbose=True, steps=args.steps, model=args.model, dataset=args.dataset, controller=args.controller)
