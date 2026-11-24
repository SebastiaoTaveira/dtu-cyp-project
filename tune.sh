#!/bin/bash
#BSUB -q hpc
#BSUB -J cyp_tune
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 02:00
#BSUB -o outputs/tune_%J.out
#BSUB -e outputs/tune_%J.err
set -uo pipefail

cd "$HOME/dtu-cyp-project"
PYTHON="$HOME/dtu-cyp-project/.DL/bin/python"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

for ov in "train.weight_decay=3.0" \
          "train.weight_decay=10.0" \
          "readout.dropout=0.3 train.weight_decay=1.0" \
          "model.readout_lr=3e-4" \
          "model.readout_lr=1e-4" \
          "train.batch_size=256 model.readout_lr=2e-3" \
          "train.batch_size=16" \
          "readout.num_layers=2" \
          "readout=linear"; do
  echo "===== classification | $ov"
  $PYTHON -u finetune.py model=morgan_rdkit_mlp task=classification device=cpu $ov
done

$PYTHON log_results.py --table
