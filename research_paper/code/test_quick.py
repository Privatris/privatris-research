#!/usr/bin/env python3
"""Quick test to verify the code works."""
import sys
sys.path.insert(0, 'src')

from train import train

print("Running quick test (1000 steps)...")
results = train(seed=42, verbose=True)
print(f"\nResults: {results}")
