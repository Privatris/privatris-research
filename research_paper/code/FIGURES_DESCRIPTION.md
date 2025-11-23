# Description des Figures pour le Paper PRIVATRIS

Ce document décrit les schémas et figures à inclure dans le papier pour illustrer l'architecture et les résultats.

## Figure 1: The PRIVATRIS Framework Architecture
**Description:** Un diagramme de haut niveau montrant les composants principaux et leur interaction.
**Contenu:**
- **Central Node:** The Agent (LLM Policy).
- **Inputs:** User Query, Retrieved Context (from Memory).
- **Outputs:** Response, Action.
- **Modules Connectés:**
    - **Privacy-Constrained Memory:** Flèche bidirectionnelle (Retrieve context / Store experience).
    - **Adversarial Self-Exploration (Red Team):** Flèche entrante (Provides adversarial prompts).
    - **Dual-Objective Update (CMDP):** Flèche cyclique (Updates Policy based on Utility & Safety).
**Légende:** "Overview of the PRIVATRIS framework. The agent interacts with the environment while the Red Team module generates adversarial scenarios. The Privacy-Constrained Memory ensures safe storage, and the Dual-Objective Update mechanism optimizes the policy under safety constraints."

## Figure 2: Dual-Objective Optimization Process
**Description:** Un flowchart détaillant la boucle d'apprentissage CMDP.
**Contenu:**
1.  **Start:** Agent generates trajectory $\tau$.
2.  **Evaluation:** Compute Utility Reward $R(\tau)$ and Safety Cost $C(\tau)$.
3.  **Constraint Check:** Is $C(\tau) \le d_k$?
    - **Yes:** Decrease $\lambda$ (Relax constraint).
    - **No:** Increase $\lambda$ (Tighten constraint via PID).
4.  **PPO Update:** Update Policy $\pi$ maximizing $R - \lambda C$.
5.  **Loop:** Repeat for next batch.
**Légende:** "The Dual-Objective Optimization loop. The Lagrangian multiplier $\lambda$ acts as a dynamic penalty, increasing when safety violations occur and decreasing when the agent is compliant, effectively balancing utility and safety."

## Figure 3: Privacy-Constrained Memory Workflow
**Description:** Un diagramme séquentiel montrant le traitement d'une interaction avant stockage.
**Contenu:**
1.  **Input:** Raw Interaction (User: "My SSN is 123-45").
2.  **Step 1: PII Detection (NER):** Detects "123-45" as SSN.
3.  **Step 2: Anonymization:** Replaces with `<SSN>`. Result: "My SSN is <SSN>".
4.  **Step 3: Embedding Check:** Compute vector $v$. Check distance to Sensitive Cluster $C_{sens}$.
    - If distance < $\delta$: **Reject**.
    - If distance > $\delta$: **Store** in Vector DB.
**Légende:** "The Privacy-Constrained Memory pipeline. Sensitive information is first sanitized via NER, and the resulting embedding is checked against known sensitive clusters to prevent latent privacy leakage."

## Figure 4: Longitudinal Safety Analysis (SVR over Time)
**Description:** Un graphique linéaire montrant l'évolution du Safety Violation Rate (SVR).
**Axe X:** Interaction Steps (0 to 10,000).
**Axe Y:** Safety Violation Rate (%).
**Séries:**
- **PPO-Unconstrained (Red):** Starts low (~4%), rises linearly to ~32%.
- **GPT-2-Constitutional (Blue):** Starts very low (~1%), rises slowly to ~6%.
- **PRIVATRIS (Green):** Starts low (~1%), stays flat or rises very slightly to ~3.7%.
**Légende:** "Safety Violation Rate (SVR) over 10,000 interaction steps. While the unconstrained baseline drifts significantly, PRIVATRIS maintains a stable safety profile throughout the agent's lifecycle."
