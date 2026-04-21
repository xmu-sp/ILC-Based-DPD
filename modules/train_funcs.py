import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import metrics
from typing import Dict, Any, Callable
import argparse


def net_train(log: Dict[str, Any], net: nn.Module, dataloader: DataLoader,
              optimizer: Optimizer, criterion: Callable, grad_clip_val: float, device: torch.device):
    """Train network for one epoch."""
    net.train()
    losses = []

    for features, targets in tqdm(dataloader):
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()
        loss = criterion(net(features), targets)
        loss.backward()

        if grad_clip_val:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip_val)

        optimizer.step()
        losses.append(loss.detach().item())

    log['loss'] = np.mean(losses)
    return net


def net_eval(log: Dict, net: nn.Module, dataloader: DataLoader,
             criterion: Callable, device: torch.device):
    """Evaluate network."""
    net.eval()
    losses, predictions, targets_list = [], [], []

    with torch.no_grad():
        for features, targets in tqdm(dataloader):
            features, targets = features.to(device), targets.to(device)
            outputs = net(features)

            losses.append(criterion(outputs, targets).item())
            predictions.append(outputs.cpu())
            targets_list.append(targets.cpu())

    log['loss'] = np.mean(losses)
    pred = torch.cat(predictions).numpy()
    truth = torch.cat(targets_list).numpy()

    return net, pred, truth


def calculate_metrics(args: argparse.Namespace, stat: Dict[str, Any],
                      pred: np.ndarray, truth: np.ndarray):
    """Calculate NMSE, EVM, and ACLR metrics."""
    stat['NMSE'] = metrics.NMSE(pred, truth)
    stat['EVM'] = metrics.EVM(pred, truth, args.bw_main_ch, args.n_sub_ch, args.nperseg)

    aclr_l, aclr_r = metrics.ACLR(pred, args.input_signal_fs, args.nperseg,
                                  args.bw_main_ch, args.n_sub_ch)
    stat['ACLR_L'] = aclr_l
    stat['ACLR_R'] = aclr_r
    stat['ACLR_AVG'] = (aclr_l + aclr_r) / 2

    return stat