import numpy as np


def count_net_params(net):
    """Count total number of parameters in a network."""
    return sum(p.numel() for p in net.parameters())


def get_amplitude(iq_signal):
    """Compute amplitude from I/Q signal."""
    return np.sqrt(iq_signal[:, 0] ** 2 + iq_signal[:, 1] ** 2)


def set_target_gain(input_iq, output_iq):
    """Calculate target gain as ratio of max amplitudes."""
    amp_in = np.max(get_amplitude(input_iq))
    amp_out = np.max(get_amplitude(output_iq))
    return amp_out / amp_in