import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

class ReversibleSequence(nn.Module):
    def __init__(self):
        super().__init__()