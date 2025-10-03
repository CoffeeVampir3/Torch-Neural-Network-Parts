import torch
import torch.nn as nn
import torch.nn.functional as F

# From the stochastic activations paper. Fairly simple, this does a 50/50 of either SILU for negatives or RELU but implemented without branching.
# https://arxiv.org/pdf/2509.22358
def stochastic_act_relu_silu_50_50(x, p=0.5):
    # The masked version outlined in the paper
    omega = (torch.rand_like(x) < p).type_as(x)

    # Continuous variant that might be more interesting if we decay p to 0 over the course of training
    # This would give us the same nice curves and a gradual trend toward sparsity.
    #omega = torch.rand_like(x) * p

    # Decompose into the positive half and negative half
    x_pos = F.relu(x)
    x_neg = F.relu(-x)

    # Silu just negative part: omega * (-x_neg) * sigmoid(-x_neg)
    silu_neg = -x_neg * torch.sigmoid(-x_neg)

    return x_pos + omega * silu_neg
