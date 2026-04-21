from torch import nn
import torch
from collections import OrderedDict
from typing import Optional, Tuple, Union

# Note: Assuming these imports are available in your project structure
from temporalAttention import TemporalAttention
from .smgu import sMGU


class RVTDSMGU(nn.Module):
    """
    RVTDSMGU: Recurrent Vector Temporal Delay sMGU Network.

    This model combines sliding window feature extraction with sMGU recurrent units
    for digital predistortion applications. It processes I/Q samples through:
    1. Feature extraction (amplitude, cubed amplitude)
    2. Sliding window temporal context
    3. sMGU recurrent processing
    4. Attention mechanism (commented) and output projection
    """

    def __init__(
            self,
            hidden_size: int,
            output_size: int,
            num_layers: int,
            bidirectional: bool = False,
            batch_first: bool = True,
            bias: bool = True,
            window_size: int = 6
    ):
        """
        Initialize the RVTDSMGU model.

        Args:
            hidden_size: Number of features in the hidden state
            output_size: Size of the output layer (typically 2 for I/Q)
            num_layers: Number of recurrent layers
            bidirectional: If True, becomes a bidirectional RNN (not used currently)
            batch_first: If True, input/output tensors are (batch, seq, feature)
            bias: If False, layers do not use bias weights
            window_size: Size of sliding window for temporal context
        """
        super(RVTDSMGU, self).__init__()

        # Model configuration
        self.hidden_size = hidden_size
        self.input_size = 4  # After feature extraction: [i, q, amp, amp^3]
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias
        self.window_size = window_size

        # Activation functions
        self.act = nn.Softmax(dim=-1)

        # Recurrent core: sMGU with windowed input
        self.rnn = sMGU(
            input_size=self.input_size * window_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

        # Fully connected layers
        # Note: fc_attention is defined but currently commented in forward pass
        self.fc_attention = nn.Linear(
            in_features=hidden_size + 2,  # +2 for sin/cos
            out_features=hidden_size,
            bias=True
        )

        self.fc_out = nn.Linear(
            in_features=hidden_size + 2,  # +2 for sin/cos
            out_features=self.output_size,
            bias=True
        )

        # Initialize parameters
        self.reset_parameters()

        # Store intermediate values for debugging
        self._debug_values = {}

    def reset_parameters(self) -> None:
        """
        Initialize model parameters.

        Note: sMGU parameter initialization is handled within the sMGU class.
        Here we only initialize the fully connected layers.
        """
        # Initialize fc_out layer
        for name, param in self.fc_out.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

        # Initialize fc_attention layer (though currently unused)
        for name, param in self.fc_attention.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract amplitude features from I/Q input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, 2) containing I/Q samples

        Returns:
            Feature tensor of shape (batch_size, seq_len, 4) containing [i, q, amp, amp^3]
        """
        # Split I and Q components
        i_x = torch.unsqueeze(x[..., 0], dim=-1)  # (batch, seq, 1)
        q_x = torch.unsqueeze(x[..., 1], dim=-1)  # (batch, seq, 1)

        # Calculate amplitude features
        amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
        amp = torch.sqrt(amp2)
        amp3 = torch.pow(amp, 3)

        # Concatenate all features
        return torch.cat((i_x, q_x, amp, amp3), dim=-1)

    def _create_sliding_windows(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create sliding windows from the input sequence and extract phase information.

        This method:
        1. Pads the input for causal windowing
        2. Creates overlapping windows of size window_size
        3. Flattens each window for sMGU input
        4. Computes phase information (sin, cos) for later use

        Args:
            x: Feature tensor of shape (batch_size, seq_len, 4)

        Returns:
            Tuple containing:
                - windows: Flattened windows (batch_size, seq_len, input_size * window_size)
                - sin: Sine of phase (batch_size, seq_len, 1)
                - cos: Cosine of phase (batch_size, seq_len, 1)
        """
        batch_size, seq_len, feat_dim = x.shape

        # Compute phase information from original I/Q (first two features are i, q)
        i_x = x[..., 0:1]  # (batch, seq, 1)
        q_x = x[..., 1:2]  # (batch, seq, 1)
        amp = torch.sqrt(torch.pow(i_x, 2) + torch.pow(q_x, 2) + 1e-8)
        sin = q_x / amp
        cos = i_x / amp

        # Pad sequence at the beginning for causal windows
        pad = x[:, -(self.window_size - 1):, :]  # Use last (window_size-1) samples as padding
        x_padded = torch.cat((pad, x), dim=1)

        # Create sliding windows
        # unfold creates windows of size window_size with step 1
        windows = x_padded.unfold(dimension=1, size=self.window_size, step=1)
        # windows shape: (batch, seq_len, feat_dim, window_size)

        # Transpose to get (batch, seq_len, window_size, feat_dim) then flatten
        windows = windows.transpose(2, 3).contiguous()
        windows = windows.reshape(batch_size, seq_len, -1)
        # windows shape: (batch, seq_len, feat_dim * window_size)

        return windows, sin, cos

    def forward(
            self,
            x: torch.Tensor,
            h_0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the RVTDSMGU model.

        The model processes input through:
        1. Feature extraction (amplitude features)
        2. Sliding window creation
        3. sMGU recurrent processing
        4. Output projection (with optional attention)

        Args:
            x: Input tensor of shape (batch_size, seq_len, 2) containing I/Q samples
            h_0: Initial hidden state for sMGU. If None, initialized with zeros.
                 Shape: (num_layers, batch_size, hidden_size)

        Returns:
            Output tensor of shape (batch_size, seq_len, output_size) typically (batch, seq, 2)
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: Extract amplitude features
        x_features = self._extract_features(x)  # (batch, seq, 4)

        # Step 2: Create sliding windows and get phase information
        windows, sin, cos = self._create_sliding_windows(x_features)
        # windows: (batch, seq, feat_dim * window_size)
        # sin, cos: (batch, seq, 1)

        # Step 3: Initialize hidden state if not provided
        if h_0 is None:
            device = x.device
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        # Step 4: Process through sMGU
        out, _ = self.rnn(windows, h_0)  # out: (batch, seq, hidden_size)

        # Step 5: Concatenate with phase information
        out = torch.cat((out, sin, cos), dim=-1)  # (batch, seq, hidden_size + 2)

        # Optional: Apply attention (currently commented out)
        # weight_attention = self.act(self.fc_attention(out))
        # out = out * weight_attention

        # Step 6: Output projection
        out = self.fc_out(out)  # (batch, seq, output_size)

        # Store debug values
        self._debug_values = {
            'windows': windows.detach(),
            'sin': sin.detach(),
            'cos': cos.detach(),
            'features': x_features.detach()
        }

        return out

    def get_debug_values(self) -> dict:
        """
        Get debug values from the last forward pass.

        Returns:
            Dictionary containing intermediate values for debugging/analysis
        """
        return self._debug_values

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize hidden state with zeros.

        Args:
            batch_size: Batch size
            device: Device for tensor creation

        Returns:
            Zero-initialized hidden state (num_layers, batch_size, hidden_size)
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)