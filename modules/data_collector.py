import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def load_dataset(dataset_name: str):
    """Load train/val/test CSV files from dataset folder."""
    path = os.path.join('datasets', dataset_name)

    def load_csv(name):
        return pd.read_csv(os.path.join(path, f'{name}.csv')).to_numpy()

    return (
        load_csv('train_input'), load_csv('train_output'),
        load_csv('val_input'), load_csv('val_output'),
        load_csv('test_input'), load_csv('test_output')
    )


def prepare_segments(args):
    """Split data into fixed-size segments with zero padding."""
    path = os.path.join('datasets', args.dataset_name)
    nperseg = args.nperseg

    def load_and_split(name):
        data = pd.read_csv(os.path.join(path, f'{name}.csv')).to_numpy()
        return _split_segments(data, nperseg)

    return (
        load_and_split('train_input'), load_and_split('train_output'),
        load_and_split('val_input'), load_and_split('val_output'),
        load_and_split('test_input'), load_and_split('test_output')
    )


def _split_segments(data, seg_len):
    """Split 2D array into segments of length seg_len, pad last if needed."""
    segments = []
    for i in range(0, len(data), seg_len):
        seg = data[i:i + seg_len]
        if len(seg) < seg_len:
            seg = np.vstack([seg, np.zeros((seg_len - len(seg), 2))])
        segments.append(seg)
    return np.array(segments)


def get_training_frames(segments, seq_len, stride=1):
    """Extract sliding windows from segments."""
    frames = []
    for seg in segments:
        for i in range((len(seg) - seq_len) // stride + 1):
            frames.append(seg[i * stride:i * stride + seq_len])
    return np.array(frames)


class IQSegmentDataset(Dataset):
    """Dataset that splits long sequences into fixed segments."""

    def __init__(self, features, targets, nperseg=16384):
        self.features = torch.Tensor(_split_segments(features, nperseg))
        self.targets = torch.Tensor(_split_segments(targets, nperseg))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class IQFrameDataset(Dataset):
    """Dataset that extracts sliding frames from sequences."""

    def __init__(self, features, targets, frame_length, stride=1):
        self.features = torch.Tensor(_get_frames(features, frame_length, stride))
        self.targets = torch.Tensor(_get_frames(targets, frame_length, stride))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def _get_frames(seq, frame_len, stride):
    """Extract sliding windows from a single sequence."""
    n_frames = (len(seq) - frame_len) // stride + 1
    frames = [seq[i * stride:i * stride + frame_len] for i in range(n_frames)]
    return np.stack(frames)


def _prepare_gmp_frames(X, y, frame_len, degree):
    """Prepare GMP-specific input/output tensors."""
    inputs, outputs = [], []
    X, y = torch.Tensor(X), torch.Tensor(y)

    for k in range(X.shape[0]):
        c_in = torch.complex(X[k, :, 0], X[k, :, 1])
        c_out = torch.complex(y[k, :, 0], y[k, :, 1])

        u_len = len(c_in) - frame_len
        in_mat = torch.stack([c_in[i:i + frame_len] for i in range(u_len)])

        # Build degree terms
        deg_terms = []
        for j in range(1, degree):
            for h in range(frame_len):
                term = in_mat[:u_len - frame_len] * torch.abs(in_mat[h:h + u_len - frame_len]) ** j
                deg_terms.append(term)

        deg_mat = torch.cat(deg_terms, dim=1) if deg_terms else torch.zeros(u_len - frame_len, 0, dtype=torch.complex64)
        in_full = torch.cat([in_mat[:u_len - frame_len], deg_mat], dim=1)

        inputs.append(in_full.numpy())
        outputs.append(c_out[:len(c_in) - 2 * frame_len].numpy())

    return inputs, outputs


class IQFrameDatasetGMP(Dataset):
    """Dataset for GMP model with polynomial degree terms."""

    def __init__(self, segment_dataset, frame_length, degree):
        X = torch.stack([item[0] for item in segment_dataset]).numpy()
        y = torch.stack([item[1] for item in segment_dataset]).numpy()
        self.X, self.y = _prepare_gmp_frames(X, y, frame_length, degree)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]