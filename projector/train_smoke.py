#!/usr/bin/env python3
"""Smoke-train the 7-vital numeric projector on the sample dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import VitalNextHourModel, masked_huber_loss


def standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = (values - mean.reshape((1,) * (values.ndim - 1) + (-1,))) / std.reshape((1,) * (values.ndim - 1) + (-1,))
    out = np.where(mask > 0, out, 0.0)
    return out.astype(np.float32)


def eval_loss(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, xm, y, ym in loader:
            x, xm, y, ym = x.to(device), xm.to(device), y.to(device), ym.to(device)
            pred = model(x, xm)
            loss = masked_huber_loss(pred, y, ym)
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
    return total / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', default='/home/vanila/code/EHR-Predict')
    ap.add_argument('--dataset', default='projector/artifacts/vital_dataset_sample.npz')
    ap.add_argument('--out', default='projector/artifacts/train_smoke_result.json')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    workdir = Path(args.workdir)
    data = np.load(workdir / args.dataset, allow_pickle=True)
    mean = data['mean'].astype(np.float32)
    std = data['std'].astype(np.float32)
    x_mask = data['vital_mask'].astype(np.float32)
    y_mask = data['target_mask'].astype(np.float32)
    x = standardize(data['vital_values'].astype(np.float32), mean, std, x_mask)
    y = standardize(data['target_values'].astype(np.float32), mean, std, y_mask)
    split = data['split']
    vital_order = [str(v) for v in data['vital_order'].tolist()]

    device = torch.device(args.device)
    torch.manual_seed(20260616)
    model = VitalNextHourModel(n_vitals=7, history_len=x.shape[1], d_model=64, n_layers=2, nhead=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def make_loader(which: int, shuffle=False):
        idx = np.where(split == which)[0]
        ds = TensorDataset(
            torch.from_numpy(x[idx]),
            torch.from_numpy(x_mask[idx]),
            torch.from_numpy(y[idx]),
            torch.from_numpy(y_mask[idx]),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle), len(idx)

    train_loader, n_train = make_loader(0, shuffle=True)
    val_loader, n_val = make_loader(1, shuffle=False)
    test_loader, n_test = make_loader(2, shuffle=False)

    history = []
    initial_val = eval_loss(model, val_loader, device) if n_val else None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_n = 0
        for xb, mb, yb, ymb in train_loader:
            xb, mb, yb, ymb = xb.to(device), mb.to(device), yb.to(device), ymb.to(device)
            pred = model(xb, mb)
            loss = masked_huber_loss(pred, yb, ymb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_total += float(loss.item()) * xb.size(0)
            train_n += xb.size(0)
        train_loss = train_total / max(1, train_n)
        val_loss = eval_loss(model, val_loader, device) if n_val else None
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

    test_loss = eval_loss(model, test_loader, device) if n_test else None

    result = {
        'dataset': str(workdir / args.dataset),
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'input_shape': list(x.shape),
        'target_shape': list(y.shape),
        'vital_order': vital_order,
        'input_observed_rate_by_vital': {vital_order[j]: float(x_mask[:, :, j].mean()) for j in range(7)},
        'target_observed_rate_by_vital': {vital_order[j]: float(y_mask[:, j].mean()) for j in range(7)},
        'initial_val_loss': initial_val,
        'final_val_loss': history[-1]['val_loss'] if history else None,
        'test_loss': test_loss,
        'history': history,
        'model': {
            'projector': 'field_emb + time_emb + mask_emb + masked value_mlp -> hour pooling',
            'encoder': '2-layer TransformerEncoder',
            'head': 'next-hour 7-vital regression',
            'loss': 'masked Huber on standardized targets',
        },
        'status': 'ok',
    }
    out = workdir / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
