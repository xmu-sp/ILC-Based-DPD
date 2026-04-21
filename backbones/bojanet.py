import torch
from torch import nn


class BOJANET(nn.Module):
    def __init__(self, hidden_size, output_size, bias=True):
        super(BOJANET, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.window_size = 16  # M in the figure
        self.num_vd_units = 6  # P in the figure
        self.bias = bias

        # FIR Filters Layer parameters
        self.fir_I = nn.Linear(self.window_size, self.num_vd_units, bias=False)
        self.fir_Q = nn.Linear(self.window_size, self.num_vd_units, bias=False)

        # Recursive Topology Layer
        vd_input_size = self.num_vd_units * 2
        self.W_fi = nn.Linear(vd_input_size, hidden_size, bias=bias)
        self.W_fh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_gi = nn.Linear(vd_input_size, hidden_size, bias=bias)
        self.W_gh = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output Layer parameters
        self.W_out_I = nn.Linear(hidden_size, 1, bias=bias)
        self.W_out_Q = nn.Linear(hidden_size, 1, bias=bias)

        self.reset_parameters()

    @staticmethod
    def vd_module(I, Q):
        """Vector Demodulator implementation."""
        magnitude = torch.sqrt(torch.pow(I, 2) + torch.pow(Q, 2))
        epsilon = 1e-8
        magnitude = magnitude + epsilon
        magnitude_squared = magnitude ** 2
        sin_theta = Q / magnitude
        cos_theta = I / magnitude
        return magnitude, magnitude_squared, sin_theta, cos_theta

    def pr_block(self, h, sin_theta, cos_theta):
        """Phase Rotation Block as shown in Fig. 10."""
        vd_units = self.num_vd_units
        hidden_size = self.hidden_size

        if vd_units >= hidden_size:
            cos_theta = cos_theta[:, :, :hidden_size]
            sin_theta = sin_theta[:, :, :hidden_size]
        elif hidden_size > vd_units and hidden_size <= vd_units * 2:
            cos_theta = torch.cat([cos_theta, cos_theta[:, :, :hidden_size - vd_units]], dim=-1)
            sin_theta = torch.cat([sin_theta, sin_theta[:, :, :hidden_size - vd_units]], dim=-1)
        elif hidden_size > vd_units * 2:
            extra_units = hidden_size - 2 * vd_units
            cos_theta = torch.cat([cos_theta, cos_theta, cos_theta[:, :, :extra_units]], dim=-1)
            sin_theta = torch.cat([sin_theta, sin_theta, sin_theta[:, :, :extra_units]], dim=-1)

        return h * cos_theta, h * sin_theta

    def _extract_windows(self, x):
        """Split input into overlapping windows for FIR filtering."""
        batch_size, seq_len, feature_size = x.shape

        # Zero padding at the beginning
        pad = torch.zeros_like(x[:, -(self.window_size - 1):, :])
        x_padded = torch.cat((pad, x), dim=1)

        # Create sliding windows
        windows = x_padded.unfold(dimension=1, size=self.window_size, step=1).transpose(2, 3)
        windows = windows.unsqueeze(2)  # (batch_size, n_windows, 1, window_size, feature_size)
        windows = windows.contiguous().view(-1, seq_len, self.window_size, feature_size)
        return windows

    def _apply_fir_filters(self, windows):
        """Apply FIR filters to I and Q components."""
        I_component = windows[:, :, :, 0]
        Q_component = windows[:, :, :, 1]

        I_fir = self.fir_I(I_component) - self.fir_Q(Q_component)
        I_fir = I_fir.contiguous().view(-1, windows.size(1), self.num_vd_units)

        Q_fir = self.fir_Q(I_component) + self.fir_I(Q_component)
        Q_fir = Q_fir.contiguous().view(-1, windows.size(1), self.num_vd_units)

        return I_fir, Q_fir

    def _recurrent_step(self, L_n, h_n):
        """Single step of the recurrent topology layer."""
        f_n = torch.sigmoid(self.W_fi(L_n) + self.W_fh(h_n))
        g_n = torch.tanh(self.W_gi(L_n) + self.W_gh(h_n))
        return f_n * h_n + (1 - f_n) * g_n

    def _process_recurrent_sequence(self, L, h_init):
        """Process the entire sequence through the recurrent layer."""
        seq_len = L.size(1)
        h_n = h_init
        h_seq = []

        for t in range(seq_len):
            L_n = L[:, t, :]
            h_n = self._recurrent_step(L_n, h_n)
            h_seq.append(h_n)

        return torch.stack(h_seq, dim=1)

    def forward(self, x, h_0=None):
        """
        Forward pass of BOJANET.

        Args:
            x: Input tensor of shape (batch_size, seq_len, 2)
            h_0: Hidden state. Can be None, (batch_size, hidden_size),
                 or (num_layers, batch_size, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, 2)
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state
        h_n = self._get_initial_hidden_state(h_0, batch_size, x.device)

        # Extract windows and apply FIR filters
        windows = self._extract_windows(x)
        I_fir, Q_fir = self._apply_fir_filters(windows)

        # Vector demodulation
        magnitude, magnitude_squared, sin_theta, cos_theta = self.vd_module(I_fir, Q_fir)

        # Prepare L for recurrent layer
        L = torch.stack([magnitude, magnitude_squared], dim=2)
        L = L.view(-1, seq_len, self.num_vd_units * 2)

        # Recurrent processing
        h_seq = self._process_recurrent_sequence(L, h_n)
        h_seq = h_seq.view(-1, seq_len, self.hidden_size)

        # Phase rotation and output
        I_rot, Q_rot = self.pr_block(h_seq, sin_theta, cos_theta)
        out_I = self.W_out_I(I_rot) - self.W_out_Q(Q_rot)
        out_Q = self.W_out_Q(Q_rot) + self.W_out_I(I_rot)

        return torch.cat([out_I, out_Q], dim=-1)

    def _get_initial_hidden_state(self, h_0, batch_size, device):
        """Extract or create the initial hidden state."""
        if h_0 is None:
            return self.init_hidden(batch_size, device)
        elif h_0.dim() == 3:  # (num_layers, batch_size, hidden_size)
            return h_0[0]  # Take first layer
        else:  # (batch_size, hidden_size)
            return h_0

    def reset_parameters(self):
        """Initialize all model parameters with appropriate strategies."""
        # FIR filter layers (small weights)
        for module in [self.fir_I, self.fir_Q]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)

        # Input-to-hidden weights (Xavier uniform)
        for module in [self.W_fi, self.W_gi]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)

        # Hidden-to-hidden weights (orthogonal for RNN stability)
        for module in [self.W_fh, self.W_gh]:
            if hasattr(module, 'weight'):
                nn.init.orthogonal_(module.weight, gain=1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)

        # Output layers (Xavier uniform)
        for module in [self.W_out_I, self.W_out_Q]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def init_hidden(self, batch_size, device):
        """Initialize hidden state with zeros."""
        return torch.zeros(batch_size, self.hidden_size, device=device)