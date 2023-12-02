import torch
import torch.nn as nn
import numpy as np
import json
from .util import (
    partition_into_batches, k_nearest_neighbors, inverse_distance_sum
)
torch.set_default_dtype(torch.float32)

class ConditionalRBM(nn.Module):

    def __init__(self):
        pass



