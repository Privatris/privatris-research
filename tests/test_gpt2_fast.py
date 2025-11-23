"""Quick test with GPT-2 (faster than Qwen on CPU)"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
import time
import torch
from src.agent import RLAgent

print("=" * 80)
print("TEST RAPIDE GPT-2 (plus rapide que Qwen sur CPU)")
print("=" * 80)

# Initialize agent with GPT-2 instead of Qwen
print("\n[1/4] Chargement GPT-2...")
start = time.time()
agent = RLAgent(
    state_dim=768,  # GPT-2 embedding dim
    action_dim=4,
    model_name='gpt2',  # 124M params vs Qwen's 498M
    device='cpu'
)
print(f"✅ Agent chargé en {time.time() - start:.1f}s")

# Test generation
print("\n[2/4] Test génération...")
test_context = "User asks: What is your name?"
start = time.time()
response = agent.policy.generate_response(test_context, max_new_tokens=30)
elapsed = time.time() - start
print(f"✅ Réponse générée en {elapsed:.1f}s")
print(f"   Contexte: {test_context}")
print(f"   Réponse: {response}")

# Test action
print("\n[3/4] Test action...")
test_state = torch.randn(768)
start = time.time()
action, log_prob, response = agent.act(test_state, test_context)
elapsed = time.time() - start
print(f"✅ Action calculée en {elapsed:.1f}s")
print(f"   Action: {action}")
print(f"   Log prob: {log_prob:.4f}")

# Test update (batch)
print("\n[4/4] Test update (batch de 32)...")
start = time.time()
for i in range(32):
    agent.update(
        reward=0.5,
        state=torch.randn(768),
        action=action,
        log_prob=log_prob,
        response=response
    )
elapsed = time.time() - start
print(f"✅ Batch update en {elapsed:.1f}s")

print("\n" + "=" * 80)
print("RÉSUMÉ")
print("=" * 80)
print("✅ GPT-2 fonctionne sur CPU!")
print(f"⏱️  Génération: ~{elapsed/32:.2f}s par step (vs >60s avec Qwen)")
print("\n💡 Recommandation: Utiliser GPT-2 pour tests, Qwen pour GPU final")
