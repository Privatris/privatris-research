# PRIVATRIS: Privacy-Constrained Reinforcement Learning for LLM Agents

This repository contains the official implementation of the paper **"PRIVATRIS: A Privacy-Preserving Reinforcement Learning Framework for Mitigating Safety Drift in Self-Evolving LLM Agents"**.

## Abstract

The deployment of Large Language Models (LLMs) as autonomous agents in high-stakes environments presents a critical challenge: **Safety Drift**. While static models can be aligned via RLHF, autonomous agents that learn from interaction are prone to "forgetting" safety constraints as they optimize for task utility.

**PRIVATRIS** is a framework designed to maintain strict safety and privacy alignment in self-evolving agents. It formulates the agent's learning process as a Constrained Markov Decision Process (CMDP), solved via a dual-objective update rule that dynamically balances utility maximization with safety constraint satisfaction.

## Repository Structure

```
.
├── paper/
│   ├── paper.pdf          # The full research paper
│   ├── paper.tex          # LaTeX source code
│   └── references.bib     # Bibliography
├── src/                   # Core framework implementation
│   ├── agent.py           # PPO Agent implementation
│   ├── cmdp.py            # Constrained MDP solver (Lagrangian Relaxation)
│   ├── memory.py          # Privacy-Constrained Memory (PCM) module
│   ├── red_team.py        # Adversarial Red Team agent
│   └── train.py           # Main training logic
├── experiments/           # Reproducibility scripts
│   ├── run_ablation.py    # Script to reproduce ablation studies (PID vs GA)
│   └── ...
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Privatris/privatris-research.git
   cd privatris-research
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Agent

To train the PRIVATRIS agent on the BeaverTails safety benchmark:

```bash
python -m src.train
```

This will start the training loop, initializing the agent, the red team adversary, and the privacy-constrained memory. Training logs will be printed to stdout.

### Reproducing Experiments

To reproduce the ablation study comparing the PID controller against standard Gradient Ascent (as reported in Section 7.3 of the paper):

```bash
python experiments/run_ablation.py
```

This script runs the comparison and generates the performance metrics demonstrating the stability benefits of the PID controller.

## Dataset

PRIVATRIS is evaluated on **BeaverTails** (PKU-Alignment/BeaverTails):
- 333,751 QA pairs with safety annotations
- 14 harm categories (violence, financial crime, privacy violations, etc.)
- Training uses 30,000 samples from the 330k training split

The dataset is automatically downloaded from HuggingFace when running the training script.

## Architecture

The framework consists of three core components:

1.  **Dual-Objective Update (CMDP)**: Uses Lagrangian relaxation with a PID controller to dynamically adjust the penalty for safety violations.
2.  **Adversarial Self-Exploration**: A "Red Team" agent that proactively generates adversarial prompts to robustify the policy.
3.  **Privacy-Constrained Memory**: A retrieval mechanism that filters PII and sensitive embeddings before storage.

## Citation

If you use this code or framework in your research, please cite our paper:

```bibtex
@article{privatris2025,
  title={PRIVATRIS: A Privacy-Preserving Reinforcement Learning Framework for Mitigating Safety Drift in Self-Evolving LLM Agents},
  author={EL YAGOUBI, Faouzi},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License.
