import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) network with customizable initialization.

    This module wraps a standard LSTM layer with specialized parameter
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
        """Initialize the LSTM and output projection layers."""
        self.rnn = nn.LSTM(
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
        self._init_lstm_parameters()
        self._init_output_projection()

    def _init_lstm_parameters(self) -> None:
        """
        Initialize LSTM parameters with orthogonal and Xavier uniform initialization.

        Note: LSTM has 4 gates (input, forget, cell, output) compared to GRU's 3 gates.
        The initialization scheme remains the same, but applied to all 4 gates.

        - Biases are initialized to zero
        - Recurrent weights (weight_hh) are initialized with orthogonal initialization
        - Input weights (weight_ih) are initialized with Xavier uniform initialization
        """
        for name, param in self.rnn.named_parameters():
            # LSTM has 4 gates: input, forget, cell, output
            num_gates = int(param.shape[0] / self.hidden_size)

            if "bias" in name:
                nn.init.constant_(param, 0)

            elif "weight_hh" in name:
                # Recurrent weights - orthogonal initialization
                for gate_idx in range(num_gates):
                    gate_slice = slice(
                        gate_idx * self.hidden_size,
                        (gate_idx + 1) * self.hidden_size
                    )
                    nn.init.orthogonal_(param[gate_slice, :])

            elif "weight_ih" in name:
                # Input weights - Xavier uniform initialization
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

    def _prepare_hidden_state(
            self,
            h_0: Optional[torch.Tensor] = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare the initial hidden state for LSTM.

        LSTM requires both hidden state (h_0) and cell state (c_0).
        If h_0 is provided, it's used for both h_0 and c_0.

        Args:
            h_0: Initial hidden state tensor

        Returns:
            Tuple of (h_0, c_0) for LSTM or None if h_0 is None
        """
        if h_0 is not None:
            return (h_0, h_0)
        return None

    def forward(
            self,
            x: torch.Tensor,
            h_0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the LSTM network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size) if batch_first=True
               else (seq_len, batch_size, input_size)
            h_0: Initial hidden state of shape (num_layers * num_directions, batch_size, hidden_size)
                 If provided, used for both hidden and cell states. If None, defaults to zeros

        Returns:
            Output tensor after LSTM processing and linear projection
        """
        # Prepare hidden state for LSTM (needs both h and c)
        hidden_state = self._prepare_hidden_state(h_0)

        # LSTM forward pass
        rnn_output, (final_hidden, final_cell) = self.rnn(x, hidden_state)

        # Output projection
        output = self.output_projection(rnn_output)

        return output

    def get_output_size(self) -> int:
        """Return the output size of the network."""
        return self.output_size

    def get_hidden_size(self) -> int:
        """Return the hidden size of the LSTM."""
        return self.hidden_size

    def get_num_directions(self) -> int:
        """Return the number of directions (1 for unidirectional, 2 for bidirectional)."""
        return 2 if self.bidirectional else 1