import inspect
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor, device, dtype, nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F


class Bottleneck(nn.Module):
    """docstring for Bottleneck"""
    def __init__(self, input_size, hidden_size, init_scale=1e-3):
        super().__init__()
        w1 = torch.empty(input_size, hidden_size)
        nn.init.normal_(w1, std=init_scale)
        self.weight1 = nn.Parameter(w1)
        self.bias1 = nn.Parameter(torch.zeros(hidden_size))

        self.activation = nn.GELU()

        w2 = torch.empty(hidden_size, input_size)
        nn.init.normal_(w2, std=init_scale)
        self.weight2 = nn.Parameter(w2)
        self.bias2 = nn.Parameter(torch.zeros(input_size))

    def forward(self, x, rel_embeds=None):
        # x: [batch, sequence, embed]
        perturbation = torch.matmul(x, self.weight1) + self.bias1
        perturbation = self.activation(perturbation)
        perturbation = torch.matmul(perturbation, self.weight2) + self.bias2

        return x + perturbation

class MultiHeadBottleneck(nn.Module):
    """docstring for Bottleneck"""
    def __init__(self, n_head, input_size, hidden_size, init_scale):
        super().__init__()
        w1 = torch.empty(1, n_head, input_size, hidden_size)
        nn.init.normal_(w1, std=init_scale)
        self.weight1 = nn.Parameter(w1)
        self.bias1 = nn.Parameter(torch.zeros(n_head, 1, hidden_size))

        self.activation = nn.GELU()

        w2 = torch.empty(1, n_head, hidden_size, input_size)
        nn.init.normal_(w2, std=init_scale)
        self.weight2 = nn.Parameter(w2)
        self.bias2 = nn.Parameter(torch.zeros(n_head, 1, input_size))

        # self.ln = nn.LayerNorm([n_head, input_size], eps=1e-05)

    def forward(self, x):
        # x: [batch, head, sequence, embed]
        perturbation = torch.matmul(x, self.weight1) + self.bias1
        perturbation = self.activation(perturbation)
        perturbation = torch.matmul(perturbation, self.weight2) + self.bias2

        # x = x + perturbation
        # x = x.permute(0, 2, 1, 3) 
        # x = self.ln(x)
        # x = x.permute(0, 2, 1, 3)

        return x + perturbation, 0.


class RelationBottleneck_Input(nn.Module):
    """docstring for Bottleneck"""
    def __init__(self, input_size, hidden_size, init_scale=1e-3):
        super().__init__()
        w1 = torch.empty(input_size+input_size, hidden_size)
        nn.init.normal_(w1, std=init_scale)
        self.weight1 = nn.Parameter(w1)
        self.bias1 = nn.Parameter(torch.zeros(hidden_size))

        self.activation = nn.GELU()

        w2 = torch.empty(hidden_size, input_size)
        nn.init.normal_(w2, std=init_scale)
        self.weight2 = nn.Parameter(w2)
        self.bias2 = nn.Parameter(torch.zeros(input_size))

    def forward(self, x, rel_embeds):
        # x: [batch, sequence, embed]
        # rel: [batch, embed]
        batch_size, seq_len, _ = x.size()
        rel_embeds = rel_embeds.view(batch_size, 1, -1).expand(-1, seq_len, -1)
        perturbation = torch.matmul(torch.cat((x, rel_embeds), dim=-1), self.weight1) + self.bias1
        perturbation = self.activation(perturbation)
        perturbation = torch.matmul(perturbation, self.weight2) + self.bias2

        return x + perturbation, 0.

class RelationBottleneck_Hyperplane(nn.Module):
    """docstring for Bottleneck"""
    def __init__(self, input_size, hidden_size, init_scale=1e-3):
        super().__init__()
        w1 = torch.empty(input_size, hidden_size)
        nn.init.normal_(w1, std=init_scale)
        self.weight1 = nn.Parameter(w1)
        self.bias1 = nn.Parameter(torch.zeros(hidden_size))

        self.activation = nn.GELU()

        w2 = torch.empty(hidden_size, input_size)
        nn.init.normal_(w2, std=init_scale)
        self.weight2 = nn.Parameter(w2)
        self.bias2 = nn.Parameter(torch.zeros(input_size))

    def projection(self, x, rel):
        norm = F.normalize(rel, p=2, dim=-1)
        return x - (x * norm).sum(dim=-1, keepdim=True) * norm 

    def forward(self, x, rel_embeds):
        # x: [batch, sequence, embed]
        # rel: [batch, embed]
        batch_size, seq_len, _ = x.size()
        rel_embeds = rel_embeds.view(batch_size, 1, -1).expand(-1, seq_len, -1)
        # x_proj = self.projection(x, rel_embeds)
        perturbation = torch.matmul(x, self.weight1) + self.bias1
        perturbation = self.activation(perturbation)
        perturbation = torch.matmul(perturbation, self.weight2) + self.bias2
        perturbation = self.projection(perturbation, rel_embeds)

        return x + perturbation, 0.

class RelationBottleneck_Projection(nn.Module):
    """docstring for Bottleneck"""
    def __init__(self, input_size, hidden_size, init_scale=1e-3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        w1 = torch.empty(input_size, hidden_size)
        nn.init.normal_(w1, std=init_scale)
        self.weight1 = nn.Parameter(w1)
        self.bias1 = nn.Parameter(torch.zeros(hidden_size))

        self.activation = nn.GELU()

        # w2 = torch.empty(hidden_size, input_size)
        # nn.init.normal_(w2, std=init_scale)
        # self.weight2 = nn.Parameter(w2)
        # self.bias2 = nn.Parameter(torch.zeros(input_size))

    def forward(self, x, rel_embeds):
        # x: [batch, sequence, embed]
        # rel: [batch, embed] --> [batch, embed1, embed2] 
        batch_size, seq_len, _ = x.size()
        rel_embeds = rel_embeds.view(batch_size, self.hidden_size, self.input_size)

        perturbation = torch.matmul(x, self.weight1) + self.bias1
        perturbation = self.activation(perturbation)
        perturbation = torch.matmul(perturbation, rel_embeds)

        return x + perturbation, 0.


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.

    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.

    Returns:
        :obj:`Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index
    
def prune_conv1d_layer(layer: Conv1D, index: torch.LongTensor, dim: int = 1) -> Conv1D:
    """
    Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights
    are transposed.

    Used to remove heads.

    Args:
        layer (:class:`~transformers.modeling_utils.Conv1D`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 1): The dimension on which to keep the indices.

    Returns:
        :class:`~transformers.modeling_utils.Conv1D`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer

