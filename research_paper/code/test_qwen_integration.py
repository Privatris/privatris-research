#!/usr/bin/env python3
"""Quick test of Qwen2.5-0.5B integration"""
import sys
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent import PrivatrisAgent, RealLLMPolicy
from memory import PrivacyConstrainedMemory

print("✓ Imports successful")

# Test 1: Create agent
print("\n[Test 1] Creating agent with Qwen2.5-0.5B...")
memory = PrivacyConstrainedMemory(embedding_dim=896)
agent = PrivatrisAgent(llm_model=None, memory=memory, learning_rate=1e-5)
print("✓ Agent created")

# Test 2: Generate response
print("\n[Test 2] Generating response...")
observation = "What is the capital of France?"
response, safety_prob, is_safe, utility_score = agent.act(observation, current_penalty=0.0, is_adversarial=False)
print(f"✓ Response generated:")
print(f"  Query: {observation}")
print(f"  Response: {response[:100]}...")
print(f"  Safety Prob: {safety_prob:.4f}")
print(f"  Is Safe: {is_safe}")
print(f"  Utility: {utility_score:.2f}")

# Test 3: Update (training step)
print("\n[Test 3] Training step...")
loss = agent.update(modified_reward=1.0, observation=observation, response=response)
print(f"✓ Update completed, loss: {loss:.4f}")

print("\n✅ All tests passed! Qwen2.5-0.5B integration working.")
