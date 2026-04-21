import torch
import torch.nn as nn


class CoreModel(nn.Module):
    """Core model factory supporting multiple backbones."""

    _BACKBONE_MAP = {
        'gmp': ('backbones.gmp', 'GMP'),
        'gru': ('backbones.gru', 'GRU'),
        'dgru': ('backbones.dgru', 'DGRU'),
        'lstm': ('backbones.lstm', 'LSTM'),
        'bojanet': ('backbones.bojanet', 'BOJANET'),
        'pgjanet': ('backbones.pgjanet', 'PGJANET'),
        'dvrjanet': ('backbones.dvrjanet', 'DVRJANET'),
        'rvtdsmgu': ('backbones.rvtdsmgu', 'RVTDSMGU'),
    }

    def __init__(self, input_size, hidden_size, num_layers, backbone_type,
                 window_size=None, num_dvr_units=None, thx=0, thh=0):
        super().__init__()
        self.backbone_type = backbone_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.thx = thx
        self.thh = thh

        self.backbone = self._build_backbone(input_size, hidden_size, num_layers,
                                             backbone_type, window_size, num_dvr_units)
        self._init_weights()

    def _build_backbone(self, input_size, hidden_size, num_layers,
                        backbone_type, window_size, num_dvr_units):
        """Factory method for creating backbone."""
        if backbone_type == 'gmp':
            from backbones.gmp import GMP
            return GMP()

        if backbone_type not in self._BACKBONE_MAP:
            raise ValueError(f"Backbone '{backbone_type}' not supported.")

        module_path, class_name = self._BACKBONE_MAP[backbone_type]
        module = __import__(module_path, fromlist=[class_name])
        backbone_cls = getattr(module, class_name)

        # Common args
        kwargs = {
            'hidden_size': hidden_size,
            'output_size': 2,
            'bias': True,
        }

        # Backbone-specific args
        if backbone_type in ['gru', 'lstm', 'rvtdsmgu']:
            kwargs.update({
                'input_size': input_size,
                'num_layers': num_layers,
                'bidirectional': False,
                'batch_first': True,
            })
        elif backbone_type == 'dgru':
            kwargs.update({'num_layers': num_layers})
        elif backbone_type == 'pgjanet':
            kwargs['window_size'] = window_size
        elif backbone_type == 'dvrjanet':
            kwargs['num_dvr_units'] = num_dvr_units

        return backbone_cls(**kwargs)

    def _init_weights(self):
        """Initialize backbone weights if method exists."""
        if hasattr(self.backbone, 'reset_parameters'):
            self.backbone.reset_parameters()
            print("Backbone Initialized...")

    def forward(self, x, h_0=None):
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        return self.backbone(x, h_0)


class CascadedModel(nn.Module):
    """Cascaded DPD + PA model."""

    def __init__(self, dpd_model, pa_model):
        super().__init__()
        self.dpd_model = dpd_model
        self.pa_model = pa_model

    def freeze_pa_model(self):
        for p in self.pa_model.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.pa_model(self.dpd_model(x))