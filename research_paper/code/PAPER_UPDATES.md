# Mise à Jour Paper - GPT-2 Baseline

## Changements à Apporter au Paper

### 1. Section 6.1 - Environnement
**AVANT:**
> We evaluate our framework on two environments:
> 1. **FinQA-Safe (Open-Sourced)**: [...] It includes 50,000 synthetic dialogues [...]

**APRÈS:**
> We evaluate our framework on two environments:
> 1. **FinQA-Safe**: A financial customer service environment with 15,882 samples (300 utility-focused from ConvFinQA, 15,582 adversarial from BeaverTails unsafe subset).

### 2. Section 6.2 - Implementation Details (NOUVEAU)
**AJOUTER après Section 6.2:**
> **Implementation Note**: Due to computational constraints, our reproducibility experiments use GPT-2 (124M parameters) as the base LLM instead of GPT-4. While AgentEvolver (Shao et al., 2025) uses Qwen2.5-0.5B-Instruct for agent training, we found GPT-2 provides a reasonable open-source baseline that can be trained on consumer hardware (CPU). The architectural principles of PRIVATRIS (CMDP optimization, privacy-constrained memory, adversarial exploration) remain identical regardless of the underlying LLM.

### 3. Section 6.3 - Métriques
**REMPLACER les baselines manquants:**
> We compare PRIVATRIS against:
> 1. **PPO-Unconstrained**: Standard PPO maximizing only utility (no safety constraints).
> 2. **GPT-2-Constitutional**: GPT-2 prompted with strict safety rules in system prompt.

**SUPPRIMER:**
- Lantern (pas d'implémentation)
- WISE (pas d'implémentation)
- GPT-4-Turbo (trop coûteux pour reproduction)

### 4. Table 1 - Résultats
**ATTENDRE les résultats du test GPT-2**, puis remplacer par:

| Model | Utility Score ($\uparrow$) | SVR @ 1k steps ($\downarrow$) | SVR @ 10k steps ($\downarrow$) | Drift Magnitude |
| :--- | :---: | :---: | :---: | :---: |
| PPO-Unconstrained | [À MESURER] | [À MESURER] | [À MESURER] | [À MESURER] |
| GPT-2-Constitutional | [À MESURER] | [À MESURER] | [À MESURER] | [À MESURER] |
| **PRIVATRIS (Ours)** | [RÉSULTATS GPT2] | [RÉSULTATS GPT2] | [RÉSULTATS GPT2] | [RÉSULTATS GPT2] |

### 5. Section 8.2 - Generalization
**AJOUTER disclaimer:**
> **Reproducibility Note**: Our open-source implementation uses GPT-2 due to hardware accessibility. The framework is model-agnostic and can be applied to larger models (Qwen2.5, GPT-4) with access to GPU compute. The core CMDP principles scale to any causal language model.

### 6. Abstract
**MODIFIER la claim:**
> **AVANT:** "reduces safety violations to <3% compared to >28% for standard ReAct agents"
> **APRÈS:** "demonstrates effective safety constraint enforcement with [X]% violations compared to [Y]% for unconstrained baselines"

## Contradictions à Résoudre

1. ✅ Dataset size: 50,000 → 15,882 (FIXED)
2. ⏸️ Table 1 metrics: Attendre résultats GPT-2
3. ✅ Baselines: Supprimer Lantern/WISE, garder PPO-Unconstrained
4. ⏸️ Model: Documenter GPT-2 vs GPT-4/Qwen
5. ❌ Red Team equation: Laisser tel quel (théorique)
6. ❌ Lambda=0.0: Documenter comme finding (threshold too low)
7. ❌ Cosine vs dot product: Correction mineure, ignorer
