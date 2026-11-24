"""Pretrained Atom-JEPA encoder with a readout head from models/readouts.py."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from atom_jepa import AtomJEPA
from models.readouts import build_readout


class Conformers(Dataset):
    def __init__(self, frame, folder, targets):
        self.frame, self.folder = frame.reset_index(drop=True), folder
        self.y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)
        #how precise each measurement was, as reported by the scientists
        sigma = frame.reindex(columns=[t + '_std' for t in targets]).to_numpy(dtype=np.float32)
        for j in range(sigma.shape[1]):
            column = sigma[:, j]
            seen = column[np.isfinite(column)]
            if seen.size == 0:
                sigma[:, j] = 1.0
                continue
            #clip to the 10th-90th percentile so a few very precise values cannot dominateand and overfit. 
            #Give molecules with no reported uncertainty the typical weight
            low, high = np.percentile(seen, 10), np.percentile(seen, 90)
            sigma[:, j] = np.where(np.isfinite(column), np.clip(column, low, high), np.median(seen))
        self.weight = (1.0 / sigma ** 2).astype(np.float32)
        print('uncertainty columns found:', int(np.isfinite(
            frame.reindex(columns=[t + '_std' for t in targets]).to_numpy(dtype=np.float32)).sum()))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        with np.load(self.folder / f'{self.frame.iloc[i].key}.npz') as d:
            return {'atomic_numbers': torch.tensor(d['z'], dtype=torch.long),
                    'node_coordinates': torch.tensor(d['pos'], dtype=torch.float32),
                    'y': torch.tensor(self.y[i]),
                    'weight': torch.tensor(self.weight[i])}


class AtomJEPAModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        self.cfg, self.folder = cfg, data_dir / 'conformers'
        self.encoder = AtomJEPA.from_pretrained(cfg.checkpoint, device=device,
                                                precision=cfg.precision, cuequivariance=False)
        self.encoder.set_grad_checkpointing(not cfg.freeze_encoder)
        self.encoder.requires_grad_(not cfg.freeze_encoder)
        self.readout = build_readout(readout, self.encoder.embedding_dim, n_targets).to(self.encoder.device)

    @property
    def device(self):
        return self.encoder.device

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval()
        return self

    def fit_preprocessing(self, train):
        pass

    #we take out the weights for the encodar not to see them and then we give them back
    def loader(self, frame, targets, batch_size, shuffle=False):
        def collate(samples):
            weight = torch.stack([s.pop('weight') for s in samples])
            batch = self.encoder.collate(samples)
            batch['weight'] = weight
            return batch

        return DataLoader(Conformers(frame, self.folder, targets), batch_size=batch_size,
                          shuffle=shuffle, collate_fn=collate, num_workers=0)

    def param_groups(self):
        groups = [{'params': list(self.readout.parameters()), 'lr': self.cfg.readout_lr}]
        if not self.cfg.freeze_encoder:
            groups.append({'params': list(self.encoder.parameters()), 'lr': self.cfg.encoder_lr})
        return groups

    # two phase freezing: first freeze encoder, then unfreeze it for fine-tuning
    def unfreeze_encoder(self):
        self.cfg.freeze_encoder = False
        self.encoder.requires_grad_(True)
        self.encoder.set_grad_checkpointing(True) 

    def forward(self, batch):
        return self.readout(self.encoder(batch))  # Differentiable forward; do NOT use encoder.embed().
