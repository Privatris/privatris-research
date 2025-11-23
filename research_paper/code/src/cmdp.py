import torch
import torch.nn as nn
import numpy as np

class LagrangianPID:
    """
    PID Controller for Lagrangian Multiplier updates to solve CMDPs.
    Minimizes the Lagrangian: L(pi, lambda) = J(pi) - lambda * (J_c(pi) - d)
    
    As described in the PRIVATRIS paper, this controller dynamically adjusts
    the penalty coefficient lambda based on the safety violation error.
    """
    def __init__(self, cost_limit: float, kp: float = 0.1, ki: float = 0.01, kd: float = 0.0, lambda_init: float = 0.0):
        """
        Args:
            cost_limit (float): The maximum allowable safety violation rate (d_k).
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
            lambda_init (float): Initial value for the Lagrange multiplier.
        """
        self.cost_limit = cost_limit
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.lambda_param = lambda_init
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, current_cost: float) -> float:
        """
        Update lambda based on the constraint violation (error).
        Error = Current Cost - Cost Limit
        If Error > 0, we are violating, so Lambda should increase (penalty increases).
        
        Args:
            current_cost (float): The current measured safety violation rate.
            
        Returns:
            float: The updated Lagrangian multiplier lambda.
        """
        error = current_cost - self.cost_limit
        
        # Integral term
        self.integral += error
        
        # Derivative term
        derivative = error - self.prev_error
        self.prev_error = error
        
        # PID update
        delta = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Update lambda (projected to >= 0)
        self.lambda_param = max(0.0, self.lambda_param + delta)
        
        return self.lambda_param

    def get_penalty(self) -> float:
        """Return the current value of lambda."""
        return self.lambda_param

class DualObjectivePPO:
    """
    Wrapper for PPO that modifies the reward function based on the Lagrangian multiplier.
    Implements the dual-objective update rule: r' = r - lambda * c
    """
    def __init__(self, lagrangian_controller: LagrangianPID):
        self.lagrangian = lagrangian_controller

    def compute_reward(self, task_reward: float, safety_cost: float) -> float:
        """
        Compute the modified reward for the policy update.
        
        Args:
            task_reward (float): The utility reward from the environment.
            safety_cost (float): The safety cost (1 if violation, 0 otherwise).
            
        Returns:
            float: The modified reward r'(s, a).
        """
        penalty = self.lagrangian.get_penalty()
        modified_reward = task_reward - (penalty * safety_cost)
        return modified_reward
