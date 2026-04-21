import warnings
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from rich.columns import Columns


class PandasLogger:
    """Logger with rich console output and CSV logging."""

    def __init__(self, path_save_file_best: str, path_log_file_hist: str,
                 path_log_file_best: str, precision: int = 8):
        self.path_save_file_best = path_save_file_best
        self.path_log_file_hist = path_log_file_hist
        self.path_log_file_best = path_log_file_best
        self.precision = precision
        self.best_val_metric = None
        self.console = Console()
        self.headers = []
        self.rows = []

    def add_row(self, headers, values):
        self.headers = headers
        row = dict(zip(headers, values))
        self.rows.append(row)
        self._display_table(row)

    def _display_table(self, stats):
        # Categorize metrics
        categories = {
            'general': [], 'train': [], 'val': [], 'test': []
        }
        for key in stats:
            if key.startswith('TRAIN_') or key == 'TRAIN_LOSS':
                categories['train'].append(key)
            elif key.startswith('VAL_'):
                categories['val'].append(key)
            elif key.startswith('TEST_'):
                categories['test'].append(key)
            else:
                categories['general'].append(key)

        # Build tables
        table_gen = Table(title="General")
        table_met = Table(title="Metrics")
        table_gen.add_column("Metric", style="cyan")
        table_gen.add_column("Value", style="green")
        table_met.add_column("Metric", style="cyan")
        table_met.add_column("Value", style="green")

        # Format value
        def fmt(v, is_lr=False):
            if isinstance(v, float):
                return f"{v:.8f}" if is_lr or abs(v) < 0.01 else f"{v:.{self.precision}f}"
            return str(v)

        # Populate tables
        for k in sorted(categories['general']):
            table_gen.add_row(k, fmt(stats[k], k == 'LR'))

        for k in sorted(categories['train']):
            table_met.add_row(k, fmt(stats[k]), style="magenta")
        for k in sorted(categories['val']):
            table_met.add_row(k, fmt(stats[k]), style="blue")
        for k in sorted(categories['test']):
            table_met.add_row(k, fmt(stats[k]), style="red")

        self.console.print(Columns([table_gen, table_met]))

    def _to_csv(self, rows, path):
        if not self.headers:
            warnings.warn("No headers defined. Call add_row first.", RuntimeWarning)
            return
        df = pd.DataFrame(rows, columns=self.headers)
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].round(self.precision)
        df.to_csv(path, index=False)

    def write_csv(self, path=None):
        self._to_csv(self.rows, path or self.path_log_file_hist)

    def write_log(self, stats):
        headers = list(stats.keys())
        values = [f"{v:.{self.precision}f}" if isinstance(v, float) else v for v in stats.values()]
        self.add_row(headers, values)
        self.write_csv()

    def write_log_idx(self, idx, path=None):
        if idx < len(self.rows):
            self._to_csv([self.rows[idx]], path or self.path_log_file_hist)

    def save_best_model(self, model, epoch, val_stats, metric='ACPR_AVG'):
        current = val_stats[metric]

        if epoch == 0 or current < self.best_val_metric:
            prev = self.best_val_metric if epoch > 0 else current
            self.best_val_metric = current
            torch.save(model.state_dict(), self.path_save_file_best)
            self.write_log_idx(epoch, self.path_log_file_best)
            self.console.print(
                f'[bold green]>>> saving best model ({prev} -> {current} {metric}) '
                f'from epoch {epoch} to {self.path_save_file_best}[/bold green]'
            )