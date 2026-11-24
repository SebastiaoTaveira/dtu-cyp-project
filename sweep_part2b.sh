#!/bin/bash
# Regression: refine around dropout=0.3 + wd=1e-1 (0.726)
REG=(
  "readout.dropout=0.2 train.weight_decay=1e-1"
  "readout.dropout=0.4 train.weight_decay=1e-1"
  "readout.dropout=0.3 train.weight_decay=3e-2"
  "readout.dropout=0.3 train.weight_decay=3e-1"
  "readout.dropout=0.3 train.weight_decay=1e-1 train.epochs=100"
  "readout.dropout=0.3 train.weight_decay=1e-1 readout.num_layers=2"
  "readout.dropout=0.3 train.weight_decay=1e-1 readout.hidden_dim=1024"
)
# Classification: refine around num_layers=4 (0.310)
CLS=(
  "readout.num_layers=4 train.weight_decay=1e-3"
  "readout.num_layers=4 train.weight_decay=1e-1"
  "readout.num_layers=4 readout.dropout=0.3"
  "readout.num_layers=4 readout.hidden_dim=1024"
  "readout.num_layers=5"
  "readout.num_layers=4 train.epochs=100"
)
for cfg in "${REG[@]}"; do
  echo "===== REG $cfg ====="
  python finetune.py model=morgan_rdkit_mlp task=regression seed=42 device=cpu train.epochs=50 $cfg
done
for cfg in "${CLS[@]}"; do
  echo "===== CLS $cfg ====="
  python finetune.py model=morgan_rdkit_mlp task=classification seed=42 device=cpu train.epochs=50 $cfg
done
python log_results.py
