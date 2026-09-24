# Scientific Collaborator Recommendation via Hypergraph Embedding

PyTorch implementation of the Scientific Collaboration Hypergraph model.

## Project Structure

```text
├── data/                   # SCH construction and processed graph
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

Cold-start scholars have fewer than five distinct collaborators in the
training set. Data, checkpoints, and generated outputs are excluded from Git.
