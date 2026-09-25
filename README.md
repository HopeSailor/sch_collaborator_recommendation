This repository contains the official implementation of our paper. If you find this work useful for your research, please cite:

> Xiaochen Wang, Wensheng Huang, Butian Zhao, and Shijuan Li<sup>\*</sup>. [**Scientific collaborator recommendation via hypergraph embedding**](https://doi.org/10.1016/j.ipm.2025.104423). *Information Processing & Management*, 2026, 63(2): 104423.

## Introduction

We propose a novel framework for collaborator recommendation based on hypergraph representation learning. Our approach models scholars and academic metadata as nodes in a **heterogeneous Scientific Collaboration Hypergraph (SCH)**, and uses a **HyperGCN-based encoder** to capture high-order relational patterns.

The framework includes: (1) Construction of a heterogeneous hypergraph from AMiner dataset; (2) Hypergraph convolution-based representation learning; (3) Coauthorship likelihood modeling via translational scoring; (4) Explanation generation using hyperedge-type templates. Experiments demonstrate the effectiveness and interpretability of our approach over strong baselines.

## Dataset

We construct the **Scientific Collaboration Hypergraph (SCH)** from [AMiner’s publicly available subsets](https://www.aminer.cn/data), including:

- Coauthorship and citation networks (Academic Social Network)
- Gender and title labels (Web User Profiling and Scholar Profiling)

Place the required source files under `data/raw/`, update their paths in `configs/sch.yaml`, and run:

```bash
python data/build_sch.py --config configs/sch.yaml

## Project Structure

```text
├── data/                   # SCH construction
├── model/                  # HyperGCN encoder and translational scorer
├── explain/                # Template-based explanations
├── evaluate.py             # Overall and cold-start evaluation
├── train.py                # Training entry point
├── utils.py                # Metrics, model setup, and data I/O
├── configs/                # YAML configurations
└── README.md
```

## Method

The model uses nine node types and nine hyperedge types. A two-layer sparse
hypergraph encoder learns 128-dimensional node embeddings. Candidate scholars
are ranked by

```text
-||h_s + r_coauthor - h_s'||_2
```

and trained with pairwise margin ranking loss.

## How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Build the SCH:

```bash
python data/build_sch.py --config configs/sch.yaml
```

Train the model:

```bash
python train.py --config configs/sch.yaml
```

Evaluate performance:

```bash
python evaluate.py --model saved_model.pt --topk 20
```

Evaluate the cold-start subset:

```bash
python evaluate.py --model saved_model.pt --topk 10 20 50 --cold-start
```

## Reproducibility Notes

The released configuration follows the below settings: a 128-dimensional
embedding space, two hypergraph convolutional layers, Adam optimization with
a learning rate of 0.001, a batch size of 512, a ranking margin of 1.0, and
early stopping with a patience of 10 epochs.

Cold-start scholars are defined as those with fewer than five distinct
collaborators in the training set. Random seeds and data construction settings
are specified in `configs/sch.yaml`.
