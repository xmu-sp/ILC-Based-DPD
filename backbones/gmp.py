import torch
import torch.nn as nn
from typing import Tuple


class GMP(nn.Module):
    """
    Generalized Memory Polynomial (GMP) model for signal processing.

    This model implements a memory polynomial with cross-terms between
    current and delayed samples up to a specified degree and memory length.
    """

    def __init__(self, memory_length: int = 11, degree: int = 5):
        super().__init__()
        self.memory_length = memory_length
        self.degree = degree

        # Calculate weight matrix dimensions
        self.num_basis_functions = 1 + (degree - 1) * memory_length
        self.weight_shape = (1, memory_length * self.num_basis_functions)

        # Initialize learnable weights
        self.weights = nn.Parameter(torch.Tensor(*self.weight_shape))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weights)

    def _prepare_complex_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert input tensor to complex representation and add zero padding.

        Args:
            x: Input tensor of shape (batch_size, frame_length, 2)

        Returns:
            Tuple of (complex_input, amplitude) tensors
        """
        # Convert to complex tensor
        complex_input = torch.complex(x[..., 0], x[..., 1])

        # Add zero padding at the beginning for memory windows
        zero_padding = torch.zeros(
            x.size(0), self.memory_length - 1,
            dtype=complex_input.dtype, device=x.device
        )
        padded_input = torch.cat([zero_padding, complex_input], dim=1)

        # Compute amplitude
        amplitude = torch.abs(padded_input)

        return padded_input, amplitude

    def _create_memory_windows(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Create sliding windows over the time dimension.

        Args:
            tensor: Input tensor of shape (batch_size, seq_len)

        Returns:
            Windowed tensor of shape (batch_size, n_windows, memory_length)
        """
        return tensor.unfold(dimension=-1, size=self.memory_length, step=1)

    def _compute_basis_terms(
            self,
            amplitude: torch.Tensor,
            batch_size: int
    ) -> torch.Tensor:
        """
        Compute polynomial basis terms for different degrees.

        Args:
            amplitude: Amplitude tensor of shape (batch_size, seq_len)
            batch_size: Batch size

        Returns:
            Tensor of shape (batch_size, degree-1, seq_len) containing power terms
        """
        basis_terms = []
        for deg in range(1, self.degree):
            power_term = torch.pow(amplitude.unsqueeze(1), deg)
            basis_terms.append(power_term)

        return torch.cat(basis_terms, dim=1)

    def _build_input_vector(
            self,
            current_window: torch.Tensor,
            mul_term: torch.Tensor,
            batch_size: int
    ) -> torch.Tensor:
        """
        Build the input feature vector for the current timestep.

        Args:
            current_window: Current complex window of shape (batch_size, memory_length)
            mul_term: Multiplication terms of shape (batch_size, degree-1, memory_length, memory_length)
            batch_size: Batch size

        Returns:
            Concatenated input vector of shape (batch_size, memory_length * num_basis_functions)
        """
        # Reshape multiplication terms
        mul_term_flat = mul_term.reshape(batch_size, -1)

        # Concatenate with current window
        return torch.cat([current_window, mul_term_flat], dim=-1)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of GMP model.

        Args:
            x: Input tensor of shape (batch_size, frame_length, 2)
            h_0: Unused parameter (kept for API compatibility)

        Returns:
            Output tensor of shape (batch_size, frame_length, 2)
        """
        batch_size, frame_length, _ = x.shape
        device = x.device

        # Initialize output tensor
        output = torch.zeros(batch_size, frame_length, 2, device=device)

        # Prepare input data
        padded_input, amplitude = self._prepare_complex_input(x)

        # Create memory windows for complex input
        complex_windows = self._create_memory_windows(padded_input)
        # Shape: (batch_size, n_windows, memory_length)

        # Prepare windows for broadcasting
        complex_windows_expanded = complex_windows.unsqueeze(1).unsqueeze(1)
        complex_windows_expanded = complex_windows_expanded.expand(
            batch_size, self.degree - 1, self.memory_length, frame_length, self.memory_length
        )

        # Compute polynomial basis terms
        basis_terms = self._compute_basis_terms(amplitude, batch_size)

        # Create memory windows for basis terms
        basis_windows = self._create_memory_windows(basis_terms)
        # Shape: (batch_size, degree-1, n_windows, memory_length)

        # Process each timestep
        for t in range(frame_length):
            # Get current complex window
            current_complex = complex_windows_expanded[:, 0, 0, t, :]

            # Get relevant basis windows for current timestep
            relevant_basis = basis_windows[:, :, t:t + self.memory_length, :]

            # Compute multiplication terms (cross-terms)
            multiplication_terms = torch.mul(
                complex_windows_expanded[:, :, :, t, :],
                relevant_basis
            )

            # Build input feature vector
            input_vector = self._build_input_vector(
                current_complex, multiplication_terms, batch_size
            )

            # Forward propagation (linear combination)
            complex_output = torch.sum(input_vector * self.weights, dim=-1)

            # Store real and imaginary parts
            output[:, t, 0] = torch.real(complex_output)
            output[:, t, 1] = torch.imag(complex_output)

        return output