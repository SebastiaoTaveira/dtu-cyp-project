"""Pretrained CheMeleon D-MPNN encoder with a readout head from models/readouts.py.

Checkpoint: chemeleon_mp.pt (Zenodo record 15460715), the message-passing encoder
from Burns et al., pretrained on 1M PubChem molecules to regress 1613 Mordred
descriptors. Input is SMILES -> 2D molecular graph; no conformers are required.
"""
import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch.utils.data import DataLoader, Dataset

from chemprop import featurizers
from chemprop import nn as cnn
from chemprop.data import BatchMolGraph

from models.readouts import build_readout


class Graphs(Dataset):
    def __init__(self, frame, featurizer, smiles_col, targets):
        self.y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)
        self.graphs = []
        for smiles in frame[smiles_col]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f'Invalid SMILES: {smiles}')
            self.graphs.append(featurizer(mol))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return self.graphs[i], self.y[i]


def collate(items):
    graphs, y = zip(*items)
    # y is built from the dataframe, so NaNs survive for masked_loss in finetune.py.
    return {'bmg': BatchMolGraph(list(graphs)), 'y': torch.from_numpy(np.stack(y))}


class ChemeleonModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        self.cfg = cfg
        checkpoint = torch.load(cfg.checkpoint, weights_only=True, map_location='cpu')
        self.encoder = cnn.BondMessagePassing(**checkpoint['hyper_parameters'])
        self.encoder.load_state_dict(checkpoint['state_dict'])
        self.aggregation = cnn.MeanAggregation()
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self.frozen = bool(cfg.freeze_encoder)
        self.encoder.requires_grad_(not self.frozen)
        self.readout = build_readout(readout, self.encoder.output_dim, n_targets)
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def fit_preprocessing(self, train):
        pass

    def loader(self, frame, targets, batch_size, shuffle=False):
        dataset = Graphs(frame, self.featurizer, self.cfg.smiles_col, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate, num_workers=0)

    def param_groups(self):
        groups = [{'params': list(self.readout.parameters()), 'lr': self.cfg.readout_lr}]
        if not self.frozen:
            groups.append({'params': list(self.encoder.parameters()), 'lr': self.cfg.encoder_lr})
        return groups

    def unfreeze_encoder(self):
        self.frozen = False
        self.encoder.requires_grad_(True)

    def forward(self, batch):
        bmg = batch['bmg']
        bmg.to(self.device)
        return self.readout(self.aggregation(self.encoder(bmg), bmg.batch))