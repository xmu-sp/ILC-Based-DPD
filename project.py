import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from arguments import get_arguments
from modules.paths import create_folder, gen_log_stat, gen_dir_paths, gen_file_paths
from modules.train_funcs import net_train, net_eval, calculate_metrics
from modules.loggers import PandasLogger


class Project:
    def __init__(self):
        self.log_all = {}
        self.log_train = {}
        self.log_val = {}
        self.log_test = {}

        self.args = get_arguments()
        self.hparams = vars(self.args)
        for k, v in self.hparams.items():
            setattr(self, k, v)

        self.load_spec()
        self._set_reproducibility()

        dir_paths = gen_dir_paths(self.args)
        self.path_dir_save, self.path_dir_log_hist, self.path_dir_log_best = dir_paths
        create_folder(dir_paths)

    def load_spec(self):
        """Load dataset specifications from JSON."""
        with open(os.path.join('datasets', self.dataset_name, 'spec.json')) as f:
            for k, v in json.load(f).items():
                setattr(self, k)
                self.hparams[k] = v

    def _set_reproducibility(self):
        """Set random seeds and deterministic behavior."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        deterministic = self.re_level == 'hard'
        torch.use_deterministic_algorithms(mode=deterministic)
        torch.backends.cudnn.benchmark = not deterministic
        torch.cuda.empty_cache()
        print(f"::: Deterministic Algorithms: {torch.are_deterministic_algorithms_enabled()}")

    def set_device(self):
        """Configure and set computing device."""
        if self.accelerator == 'cuda' and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.devices}")
            torch.cuda.set_device(device)
            print(f"::: GPU {self.devices}: {torch.cuda.get_device_name(self.devices)}")
        elif self.accelerator == 'mps' and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif self.accelerator == 'cpu':
            device = torch.device("cpu")
        else:
            raise ValueError(f"Device '{self.accelerator}' not supported.")

        self.add_arg("device", device)
        return device

    def add_arg(self, key, value):
        """Add runtime argument."""
        setattr(self, key, value)
        self.hparams[key] = value

    def _gen_model_id(self, prefix, backbone, hidden_size, n_params, extra=None):
        """Generate model identifier string."""
        parts = ['S', str(self.seed), 'M', backbone.upper(),
                 'H', str(hidden_size), 'F', str(self.frame_length), 'P', str(n_params)]
        if extra:
            parts.extend(extra)
        return f"{prefix}_{'_'.join(parts)}"

    def gen_pa_model_id(self, n_params):
        return self._gen_model_id('PA', self.PA_backbone, self.PA_hidden_size, n_params)

    def gen_dpd_model_id(self, n_params):
        extra = []
        if 'delta' in self.DPD_backbone:
            extra.extend(['THX', f"{self.thx:.3f}", 'THH', f"{self.thh:.3f}"])
        return self._gen_model_id('DPD', self.DPD_backbone, self.DPD_hidden_size, n_params, extra)

    def build_logger(self, model_id):
        paths = gen_file_paths(self.path_dir_save, self.path_dir_log_hist,
                               self.path_dir_log_best, model_id)
        self.path_save_file_best, self.path_log_file_hist, self.path_log_file_best = paths
        for name, path in zip(['Save', 'History', 'Best'], paths):
            print(f"::: {name:7} Path: {path}")

        self.logger = PandasLogger(*paths, precision=self.log_precision)

    def build_dataloaders(self):
        from modules.data_collector import IQSegmentDataset, IQFrameDataset, load_dataset

        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(self.dataset_name)

        if self.step == 'train_dpd':
            self.target_gain = np.max(util.get_amplitude(y_train)) / np.max(util.get_amplitude(X_train))
            y_train = self.target_gain * X_train
            y_val = self.target_gain * X_val
            y_test = self.target_gain * X_test

        train_set = IQFrameDataset(X_train, y_train, self.frame_length, self.frame_stride)
        val_set = IQSegmentDataset(X_val, y_val, self.args.nperseg)
        test_set = IQSegmentDataset(X_test, y_test, self.args.nperseg)

        train_loader = DataLoader(train_set, self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, self.batch_size_eval, shuffle=False)
        test_loader = DataLoader(test_set, self.batch_size_eval, shuffle=False)

        return (train_loader, val_loader, test_loader), X_train.shape[-1]

    def build_criterion(self):
        criteria = {'l2': nn.MSELoss(), 'l1': nn.L1Loss()}
        if self.loss_type not in criteria:
            raise ValueError(f"Loss '{self.loss_type}' not supported.")
        return criteria[self.loss_type]

    def build_optimizer(self, net):
        optimizers = {
            'adam': optim.Adam, 'sgd': lambda p: optim.SGD(p, lr=self.lr, momentum=0.9),
            'rmsprop': optim.RMSprop, 'adamw': optim.AdamW
        }
        if self.opt_type not in optimizers and self.opt_type != 'adabound':
            raise ValueError(f"Optimizer '{self.opt_type}' not supported.")

        if self.opt_type == 'adabound':
            import adabound
            optimizer = adabound.AdaBound(net.parameters(), lr=self.lr, final_lr=0.1)
        else:
            optimizer = optimizers[self.opt_type](net.parameters(), lr=self.lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.decay_factor, patience=self.patience,
            verbose=True, threshold=1e-4, min_lr=self.lr_end
        )
        return optimizer, scheduler

    def train(self, net, criterion, optimizer, scheduler, train_loader, val_loader, test_loader, best_metric):
        start = time.time()

        for epoch in range(self.n_epochs):
            net = net_train(self.log_train, net, train_loader, optimizer, criterion,
                            self.grad_clip_val, self.device)

            for name, loader, log in [('val', val_loader, self.log_val),
                                      ('test', test_loader, self.log_test)]:
                if getattr(self, f'eval_{name}'):
                    _, pred, truth = net_eval(log, net, loader, criterion, self.device)
                    log.update(calculate_metrics(self.args, log, pred, truth))

            elapsed = (time.time() - start) / 60
            self.log_all = gen_log_stat(self.args, elapsed, net, optimizer, epoch,
                                        self.log_train, self.log_val, self.log_test)
            self.logger.write_log(self.log_all)

            best_net = net.dpd_model if self.step == 'train_dpd' else net
            self.logger.save_best_model(best_net, epoch, self.log_val, best_metric)

            if self.lr_schedule:
                scheduler.step(self.log_val[best_metric])

        print("Training Completed.\n")