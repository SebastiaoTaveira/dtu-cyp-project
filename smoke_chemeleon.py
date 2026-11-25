"""Feasibility check before any experiments (required by the project description)."""
from pathlib import Path
import pandas as pd
import torch
from omegaconf import OmegaConf
from models.chemeleon import ChemeleonModel
from prepare_data import REG

cfg = OmegaConf.load('conf/model/chemeleon.yaml')

readout = OmegaConf.merge(OmegaConf.load('conf/readout/mlp.yaml'), cfg.get('readout', {}))

train = pd.read_csv('data/regression/train.csv')

model = ChemeleonModel(cfg.model, readout, len(REG), Path('data'), 'cpu')

print('embedding dim :', model.encoder.output_dim)

loader = model.loader(train.head(8), REG, batch_size=8)
batch = next(iter(loader))
print('y shape       :', batch['y'].shape,
      '| NaNs present:', bool(torch.isnan(batch['y']).any()))

bmg = batch['bmg']
z = model.aggregation(model.encoder(bmg), bmg.batch)
print('embedding     :', z.shape)

pred = model(batch)
print('prediction    :', pred.shape, '| finite:', bool(torch.isfinite(pred).all()))

loss = pred.sum()
loss.backward()
print('backward pass : ok')

for split in ['train', 'val', 'test']:
    frame = pd.read_csv(f'data/regression/{split}.csv')
    model.loader(frame, REG, batch_size=64)
    print(f'{split:5s} featurized: {len(frame)} molecules')