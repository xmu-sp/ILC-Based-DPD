import torch
import torch.nn as nn


class DGRU(nn.Module):
    """Deep GRU network with input feature engineering and residual connections."""

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        bidirectional: bool = False,
        batch_first: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = 6  # Fixed input feature dimension
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias

        self._build_layers()
        self.reset_parameters()

    def _build_layers(self) -> None:
        """Initialize network layers."""
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=self.batch_first,
            bias=self.bias,
        )
        self.fc_out = nn.Linear(
            in_features=self.hidden_size + self.input_size,
            out_features=self.output_size,
            bias=self.bias,
        )
        self.fc_hid = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=self.bias,
        )

    def reset_parameters(self) -> None:
        """Initialize network parameters with appropriate schemes."""
        self._init_rnn_parameters()
        self._init_fc_parameters(self.fc_out, weight_init="xavier_uniform")
        self._init_fc_parameters(self.fc_hid, weight_init="kaiming_uniform")

    def _init_rnn_parameters(self) -> None:
        """Initialize GRU layer parameters."""
        for name, param in self.rnn.named_parameters():
            num_gates = int(param.shape[0] / self.hidden_size)

            if "bias" in name:
                nn.init.constant_(param, 0)

            if "weight" in name:
                for i in range(num_gates):
                    gate_slice = slice(i * self.hidden_size, (i + 1) * self.hidden_size)
                    nn.init.orthogonal_(param[gate_slice, :])

            if "weight_ih_l0" in name:
                for i in range(num_gates):
                    gate_slice = slice(i * self.hidden_size, (i + 1) * self.hidden_size)
                    nn.init.xavier_uniform_(param[gate_slice, :])

    @staticmethod
    def _init_fc_parameters(layer: nn.Linear, weight_init: str) -> None:
        """Initialize fully connected layer parameters."""
        for name, param in layer.named_parameters():
            if "weight" in name:
                init_fn = getattr(nn.init, weight_init)
                init_fn(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract engineered features from I/Q components."""
        i_x = x[..., 0:1]
        q_x = x[..., 1:2]

        amp2 = i_x.pow(2) + q_x.pow(2)
        amp = amp2.sqrt()
        amp3 = amp.pow(3)

        cos = i_x / amp
        sin = q_x / amp

        return torch.cat([i_x, q_x, amp, amp3, sin, cos], dim=-1)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DGRU network.

        Args:
            x: Input tensor with shape (batch, seq_len, 2) containing I/Q pairs
            h_0: Initial hidden state for GRU

        Returns:
            Output tensor after processing through the network
        """
        # Feature extraction from I/Q components
        features = self._extract_features(x)

        # GRU processing
        rnn_out, _ = self.rnn(features, h_0)

        # Hidden layer with ReLU activation
        hidden_out = torch.relu(self.fc_hid(rnn_out))

        # Residual connection concatenating hidden output with input features
        combined = torch.cat([hidden_out, features], dim=-1)

        # Output projection
        output = self.fc_out(combined)

        return output