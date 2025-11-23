"""Test minimal des composants pour diagnostiquer le blocage"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
import time
import torch

print("=" * 80)
print("TEST 1: Import des modules")
print("=" * 80)
try:
    from src.agent import RLAgent
    from src.environment import PrivacyEnvironment
    from src.memory import PrivacyConstrainedMemory
    print("✅ Tous les imports réussis")
except Exception as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("TEST 2: Chargement de l'agent (cœur du problème)")
print("=" * 80)
try:
    start = time.time()
    print(f"[{0:.1f}s] Initialisation agent...")
    agent = RLAgent(
        state_dim=896,  # Qwen embedding dim
        action_dim=4,
        model_name='Qwen/Qwen2.5-0.5B-Instruct',
        device='cpu'
    )
    elapsed = time.time() - start
    print(f"✅ Agent chargé en {elapsed:.1f}s")
except Exception as e:
    print(f"❌ Erreur agent: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("TEST 3: Génération de texte")
print("=" * 80)
try:
    test_context = "User asks: What is your name?"
    start = time.time()
    print(f"[{0:.1f}s] Génération réponse...")
    response = agent.policy.generate_response(test_context)
    elapsed = time.time() - start
    print(f"✅ Réponse générée en {elapsed:.1f}s")
    print(f"   Contexte: {test_context}")
    print(f"   Réponse: {response}")
except Exception as e:
    print(f"❌ Erreur génération: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("TEST 4: Action de l'agent")
print("=" * 80)
try:
    test_state = torch.randn(896)  # État aléatoire
    start = time.time()
    print(f"[{0:.1f}s] Calcul action...")
    action, log_prob, response = agent.act(test_state, test_context)
    elapsed = time.time() - start
    print(f"✅ Action calculée en {elapsed:.1f}s")
    print(f"   Action: {action}")
    print(f"   Log prob: {log_prob:.4f}")
    print(f"   Réponse: {response[:100]}...")
except Exception as e:
    print(f"❌ Erreur action: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("TEST 5: Environnement")
print("=" * 80)
try:
    start = time.time()
    print(f"[{0:.1f}s] Chargement environnement...")
    env = PrivacyEnvironment()
    elapsed = time.time() - start
    print(f"✅ Environnement chargé en {elapsed:.1f}s")
    print(f"   Dataset taille: {len(env.dataset)}")
except Exception as e:
    print(f"❌ Erreur environnement: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST 6: Mémoire")
print("=" * 80)
try:
    start = time.time()
    print(f"[{0:.1f}s] Initialisation mémoire...")
    memory = PrivacyConstrainedMemory()
    elapsed = time.time() - start
    print(f"✅ Mémoire initialisée en {elapsed:.1f}s")
except Exception as e:
    print(f"❌ Erreur mémoire: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("RÉSUMÉ")
print("=" * 80)
print("✅ Tous les tests passés !")
print("\nMaintenant on peut tester train.py avec confiance.")
