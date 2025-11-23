# Diagnostic de Performance - Qwen2.5-0.5B sur CPU

## Problème Identifié
La génération de texte avec Qwen2.5-0.5B-Instruct est **extrêmement lente sur CPU** :
- **Chargement du modèle** : ~5 secondes ✅ (acceptable)
- **Génération de 50 tokens** : >60 secondes ❌ (bloque le training)

## Cause Racine
- Qwen2.5-0.5B (498M paramètres) en fp16 sur CPU = inférence très lente
- Pas de support MPS pour `torch.isin` (utilisé dans Qwen generate)
- Autoregressive generation fait 50 forward passes (1 token à la fois)

## Solutions Possibles

### Option 1: Utiliser un modèle plus petit (RECOMMANDÉ)
```python
# GPT-2 small (124M params) - 4x plus petit
model_name = 'gpt2'  # ou 'distilgpt2' (82M)
```
**Avantages:**
- 4-10x plus rapide
- Toujours génératif (causal LM)
- Compatibilité totale CPU/MPS

**Inconvénients:**
- Moins capable que Qwen
- Pas exactement AgentEvolver (qui utilise Qwen)

### Option 2: Optimiser Qwen pour CPU
```python
# Charger en int8 (quantization)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct',
    load_in_8bit=True,  # Réduit mémoire et accélère
    device_map="cpu"
)
```
**Avantages:**
- Garde Qwen (AgentEvolver-compliant)
- 2-3x plus rapide

**Inconvénients:**
- Requiert bitsandbytes
- Toujours lent (>20s par génération)

### Option 3: Réduire max_new_tokens
```python
# Au lieu de 50 tokens, générer seulement 20
response = policy.generate_response(context, max_new_tokens=20)
```
**Avantages:**
- 2.5x plus rapide (20s au lieu de 50s)
- Moins de ressources

**Inconvénients:**
- Réponses tronquées
- Toujours lent pour training

### Option 4: Mode "simulation" avec templates (FALLBACK)
```python
# Pour tests rapides uniquement
USE_SIMULATION = True
if USE_SIMULATION:
    # Retourner templates au lieu de générer
    if "credit card" in context.lower():
        return "I cannot help with that request."
    else:
        return "I'd be happy to help with your question."
```
**Avantages:**
- Instantané
- Permet de tester la boucle RL

**Inconvénients:**
- Pas un vrai LLM
- Inutile pour évaluation finale

## Recommandation Finale

**Pour validation immédiate:**
1. Utiliser GPT-2 (`gpt2`) pour tester la pipeline complète
2. Vérifier que PPO, memory, red-team fonctionnent
3. Obtenir des métriques SVR/Utility avec GPT-2

**Pour paper final:**
1. Documenter la limitation CPU
2. Expliquer que Qwen nécessite GPU (A100 dans AgentEvolver)
3. Proposer GPT-2 comme baseline reproductible
4. Ou: Run sur Colab/Kaggle avec GPU gratuit

## Prochaines Étapes
1. ✅ Tests composants passent (agent se charge)
2. 🔄 Choisir GPT-2 pour tests rapides
3. ⏸️ Tester training complet avec 10 steps
4. ⏸️ Comparer résultats avec paper
5. ⏸️ Mettre à jour paper.md avec vraies métriques
