import torch
import torch.nn as nn
from typing import Optional, Tuple


class GRU(nn.Module):
    """
    Gated Recurrent Unit (GRU) network with customizable initialization.

    This module wraps a standard GRU layer with specialized parameter
    initialization schemes and an output projection layer.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_layers: int,
            bidirectional: bool = False,
            batch_first: bool = True,
            bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias

        self._build_layers()
        self.reset_parameters()

    def _build_layers(self) -> None:
        """Initialize the GRU and output projection layers."""
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=self.batch_first,
            bias=self.bias,
        )

        self.output_projection = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.output_size,
            bias=True,
        )

    def reset_parameters(self) -> None:
        """Initialize network parameters with specialized schemes."""
        self._init_gru_parameters()
        self._init_output_projection()

    def _init_gru_parameters(self) -> None:
        """
        Initialize GRU parameters with orthogonal and Xavier uniform initialization.

        - Biases are initialized to zero
        - Recurrent weights are initialized with orthogonal initialization
        - Input weights are initialized with Xavier uniform initialization
        """
        for name, param in self.rnn.named_parameters():
            num_gates = int(param.shape[0] / self.hidden_size)

            if "bias" in name:
                nn.init.constant_(param, 0)

            elif "weight" in name:
                for gate_idx in range(num_gates):
                    gate_slice = slice(
                        gate_idx * self.hidden_size,
                        (gate_idx + 1) * self.hidden_size
                    )
                    nn.init.orthogonal_(param[gate_slice, :])

            elif "weight_ih_l0" in name:
                for gate_idx in range(num_gates):
                    gate_slice = slice(
                        gate_idx * self.hidden_size,
                        (gate_idx + 1) * self.hidden_size
                    )
                    nn.init.xavier_uniform_(param[gate_slice, :])

    def _init_output_projection(self) -> None:
        """
        Initialize output projection layer parameters.

        - Weights are initialized with Xavier uniform initialization
        - Biases are initialized to zero
        """
        for name, param in self.output_projection.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(
            self,
            x: torch.Tensor,
            h_0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the GRU network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size) if batch_first=True
               else (seq_len, batch_size, input_size)
            h_0: Initial hidden state of shape (num_layers * num_directions, batch_size, hidden_size)
                 If None, defaults to zeros

        Returns:
            Output tensor after GRU processing and linear projection
        """
        # GRU forward pass
        rnn_output, _ = self.rnn(x, h_0)

        # Output projection
        output = self.output_projection(rnn_output)

        return output

    def get_output_size(self) -> int:
        """Return the output size of the network."""
        return self.output_size

    def get_hidden_size(self) -> int:
        """Return the hidden size of the GRU."""
        return self.hidden_size