import torch
import torch.nn as nn
from typing import Optional, List, Tuple


class sMGUCell(nn.Module):
    """Simplified Minimal Gated Unit Cell."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.r_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.r_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.w_f, self.w_c]:
            nn.init.xavier_uniform_(w)
        for r in [self.r_f, self.r_c]:
            nn.init.orthogonal_(r)
        for b in [self.b_f, self.b_c]:
            nn.init.zeros_(b)

    def forward(self, x, states):
        h_prev, c_prev, n_prev, m_prev = states

        f_t = torch.exp(
            torch.matmul(x, self.w_f) +
            torch.matmul(h_prev, self.r_f) +
            self.b_f
        )

        c_tilda = (
                torch.matmul(x, self.w_c) +
                torch.matmul(h_prev, self.r_c) +
                self.b_c
        )

        m_t = torch.max(
            torch.log(f_t) + m_prev,
            torch.log(torch.exp(torch.zeros_like(f_t)))
        )

        i_prime = torch.exp(torch.zeros_like(f_t) - m_t)
        f_prime = torch.exp(torch.log(f_t) + m_prev - m_t)

        c_t = f_prime * c_prev + i_prime * torch.tanh(c_tilda)
        n_t = f_prime * n_prev + i_prime

        h_t = torch.tanh(c_t / n_t.clamp(min=1e-8))
        return h_t, (h_t, c_t, n_t, m_t)


class sMGU(nn.Module):
    """Simplified Minimal Gated Unit."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            sMGUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

    def forward(self, x, initial_states=None):
        batch_size, seq_len, _ = x.shape

        if initial_states is None:
            current_states = [
                (torch.zeros(batch_size, self.hidden_size, device=x.device),) * 4
                for _ in range(self.num_layers)
            ]
        else:
            current_states = initial_states

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            new_states = []
            for layer, state in zip(self.layers, current_states):
                x_t, new_state = layer(x_t, state)
                new_states.append(new_state)
            outputs.append(x_t.unsqueeze(1))
            current_states = new_states

        return torch.cat(outputs, dim=1), current_states