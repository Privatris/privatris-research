#!/usr/bin/env python3
"""Test rapide avec GPT-2 - 100 steps pour obtenir des métriques réelles"""
import os
import sys

# Add project root to path
sys.path.insert(0, '/Users/yagobski/Documents/GIA/Documents/Thesepoly/research_paper/code')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

print("=" * 80)
print("PRIVATRIS - Test GPT-2 (100 steps)")
print("=" * 80)
print()

from src.train import train

# Run training with GPT-2
results = train(seed=42, verbose=True)

print("\n" + "=" * 80)
print("RÉSULTATS FINAUX")
print("=" * 80)
print(f"SVR Initial:     {results.get('initial_svr', 0):.2%}")
print(f"SVR Final:       {results.get('final_svr', 0):.2%}")
print(f"Drift:           +{results.get('drift', 0):.2%}")
print(f"Utility Score:   {results.get('utility', 0):.2f}/10")
print(f"Lambda Final:    {results.get('lambda', 0):.4f}")
print()
print("✅ Test terminé! Vous pouvez maintenant mettre à jour paper.md avec ces métriques.")
