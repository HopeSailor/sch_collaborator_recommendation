This is the official codebase for our paper:

> **Scientific Collaborator Recommendation via Hypergraph Embedding**  
> Xiaochen Wang, Wensheng Huang, Butian Zhao, and Shijuan Li. *Information Processing & Management*, 2026, 63(2): 104423.

## Introduction

We propose a novel framework for collaborator recommendation based on hypergraph representation learning. Our approach models scholars and academic metadata as nodes in a **heterogeneous Scientific Collaboration Hypergraph (SCH)**, and uses a **HyperGCN-based encoder** to capture high-order relational patterns.

The framework includes: (1) Construction of a heterogeneous hypergraph from AMiner dataset; (2) Hypergraph convolution-based representation learning; (3) Coauthorship likelihood modeling via translational scoring; (4) Explanation generation using hyperedge-type templates. Experiments demonstrate the effectiveness and interpretability of our approach over strong baselines.

## Dataset

We construct the **Scientific Collaboration Hypergraph (SCH)** from [AMiner’s publicly available subsets](https://www.aminer.cn/data), including:

- Coauthorship and citation networks (Academic Social Network)
- Gender and title labels (Web User Profiling and Scholar Profiling)

Processed node and hyperedge lists are provided under `data/`.

## Environment Requirements

- Python 3.8+
- PyTorch ≥ 1.12
- NumPy
- NetworkX
- Scikit-learn
- tqdm
- matplotlib

```bash
conda create -n sch_env python=3.8
conda activate sch_env
pip install -r requirements.txt
```

## Project Structure
```graphql
├── data/                   # Processed SCH data
├── model/                  # HyperGCN and scoring modules
├── explain/                # Template-based explanation generation
├── evaluate.py             # Evaluation pipeline
├── train.py                # Main training script
├── utils.py                # Metric, data I/O utilities
├── configs/                # Config YAMLs
└── README.md
```

## How to Run
```bash
# Train the SCH model
python train.py --config configs/sch.yaml
# Evaluate performance
python evaluate.py --model saved_model.pt --topk 20
```

## Contact
For questions or feedback, feel free to open an issue or contact via my e-mail: xcwang25@stu.pku.edu.cn
