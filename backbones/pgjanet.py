import torch
import torch.nn as nn
from typing import Optional, Tuple


class PGJANET(nn.Module):
    """
    Phase-Gated Joint Amplitude Network (PGJANET).

    This network processes I/Q signal components using specialized gating mechanisms
    that incorporate amplitude and phase information through multiple processing paths.
    """

    def __init__(
            self,
            hidden_size: int,
            output_size: int,
            bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias = bias

        self._build_layers()
        self.reset_parameters()

    def _build_layers(self) -> None:
        """Initialize all network layers."""
        # Input gates for different signal components
        self.amplitude_gate = nn.Linear(
            self.hidden_size + 1, self.hidden_size, bias=self.bias
        )
        self.phase_cos_gate = nn.Linear(
            self.hidden_size + 1, self.hidden_size, bias=self.bias
        )
        self.phase_sin_gate = nn.Linear(
            self.hidden_size + 1, self.hidden_size, bias=self.bias
        )

        # Processing gates for state update
        self.forget_gate = nn.Linear(
            self.hidden_size * 2, self.hidden_size, bias=self.bias
        )
        self.update_gate = nn.Linear(
            self.hidden_size * 2, self.hidden_size, bias=self.bias
        )

        # Output projection
        self.output_projection = nn.Linear(
            self.hidden_size, self.output_size, bias=self.bias
        )

    def _compute_signal_features(
            self, i_component: torch.Tensor, q_component: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute amplitude and phase features from I/Q components.

        Args:
            i_component: In-phase component of shape (batch_size, 1)
            q_component: Quadrature component of shape (batch_size, 1)

        Returns:
            Tuple of (amplitude, cos_phase, sin_phase) tensors
        """
        amplitude = torch.sqrt(i_component.pow(2) + q_component.pow(2))
        phase = torch.atan2(q_component, i_component)

        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        return amplitude, cos_phase, sin_phase

    def _compute_intermediate_state(
            self,
            hidden_state: torch.Tensor,
            amplitude: torch.Tensor,
            cos_phase: torch.Tensor,
            sin_phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the intermediate state u_n through gating mechanisms.

        Args:
            hidden_state: Current hidden state of shape (batch_size, hidden_size)
            amplitude: Amplitude tensor of shape (batch_size, 1)
            cos_phase: Cosine of phase tensor of shape (batch_size, 1)
            sin_phase: Sine of phase tensor of shape (batch_size, 1)

        Returns:
            Intermediate state tensor of shape (batch_size, hidden_size)
        """
        # Concatenate hidden state with different signal components
        hidden_amp = torch.cat([hidden_state, amplitude], dim=-1)
        hidden_cos = torch.cat([hidden_state, cos_phase], dim=-1)
        hidden_sin = torch.cat([hidden_state, sin_phase], dim=-1)

        # Compute individual gate outputs
        amp_gate_out = torch.tanh(self.amplitude_gate(hidden_amp))
        cos_gate_out = torch.tanh(self.phase_cos_gate(hidden_cos))
        sin_gate_out = torch.tanh(self.phase_sin_gate(hidden_sin))

        # Compute combined intermediate state with multiplicative interactions
        # u_n = a_n * p1_n * p2_n * (1 - a_n) * (1 - p1_n) * (1 - p2_n)
        combined_product = amp_gate_out * cos_gate_out * sin_gate_out
        complement_product = (1 - amp_gate_out) * (1 - cos_gate_out) * (1 - sin_gate_out)

        return combined_product * complement_product

    def _update_hidden_state(
            self,
            hidden_state: torch.Tensor,
            intermediate_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Update hidden state using forget and update gates.

        Args:
            hidden_state: Current hidden state of shape (batch_size, hidden_size)
            intermediate_state: Intermediate state u_n of shape (batch_size, hidden_size)

        Returns:
            Updated hidden state of shape (batch_size, hidden_size)
        """
        # Concatenate hidden state with intermediate state
        hidden_combined = torch.cat([hidden_state, intermediate_state], dim=-1)

        # Compute gates
        forget_gate = torch.sigmoid(self.forget_gate(hidden_combined))
        update_value = torch.tanh(self.update_gate(hidden_combined))

        # Update hidden state: h_new = f_n * h + (1 - f_n) * g_n
        return forget_gate * hidden_state + (1 - forget_gate) * update_value

    def forward(
            self,
            x: torch.Tensor,
            h_0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of PGJANET.

        Args:
            x: Input tensor of shape (batch_size, seq_len, 2) containing I/Q pairs
            h_0: Initial hidden state of shape (num_layers, batch_size, hidden_size)
                 If None, initializes to zeros. Only the first layer's state is used.

        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden state
        if h_0 is not None:
            hidden_state = h_0[0]  # Take only the first layer's hidden state
        else:
            hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)

        outputs = []

        # Process sequence step by step
        for t in range(seq_len):
            # Extract current timestep I/Q components
            x_t = x[:, t, :]
            i_component = x_t[:, 0:1]
            q_component = x_t[:, 1:2]

            # Compute signal features
            amplitude, cos_phase, sin_phase = self._compute_signal_features(
                i_component, q_component
            )

            # Compute intermediate state
            intermediate_state = self._compute_intermediate_state(
                hidden_state, amplitude, cos_phase, sin_phase
            )

            # Update hidden state
            hidden_state = self._update_hidden_state(hidden_state, intermediate_state)

            # Generate output
            output = self.output_projection(hidden_state)
            outputs.append(output)

        # Stack outputs along sequence dimension
        return torch.stack(outputs, dim=1)

    def reset_parameters(self) -> None:
        """Initialize all network parameters using Xavier uniform initialization."""
        layers_to_init = [
            self.amplitude_gate,
            self.phase_cos_gate,
            self.phase_sin_gate,
            self.forget_gate,
            self.update_gate,
            self.output_projection,
        ]

        for layer in layers_to_init:
            if hasattr(layer, 'weight') and layer.weight is not None:
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def get_hidden_size(self) -> int:
        """Return the hidden size of the network."""
        return self.hidden_size

    def get_output_size(self) -> int:
        """Return the output size of the network."""
        return self.output_size