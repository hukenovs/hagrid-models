import torch


def _is_tracing():
    return torch._C._get_tracing_state()
