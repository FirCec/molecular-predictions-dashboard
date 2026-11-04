# Understanding Molecular Predictions

Interactive visual analytics dashboard to explore how a Graph Neural Network (GNN)
predicts the aqueous solubility of molecules (ESOL dataset), where it succeeds or
fails, and how faithful its explanations are.

Master Project "Visualization and Artificial Intelligence", RPTU, Winter 2026/27.

## Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data

`data/raw/delaney-processed.csv`: ESOL dataset (Delaney, 2004), 1,128 molecules
with measured log solubility (mol/L).