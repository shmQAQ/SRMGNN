import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric import utils
from torch_geometric.nn.aggr import Aggregation
from torch_scatter import scatter
from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from typing import Any, Optional, List, Dict, Union
from torch_geometric.nn.inits import reset
from torch_geometric.utils import degree


class SqaredReLu(nn.Module):
    '''fast implementation of squared relu function'''

    def __init__(self):
        super(SqaredReLu, self).__init__()

    def forward(self, x):
        return torch.pow(F.relu(x), 2)


class SpatialDepthWiseConv(nn.Module):
    '''
    the spatial depthwise convolution in front of q,k,v
    '''

    def __init__(self, d_k: int, kenerl_size: int = 3):
        super(SpatialDepthWiseConv, self).__init__()
        self.conv = nn.Conv1d(d_k, d_k, kenerl_size,
                              padding=kenerl_size-1, groups=d_k)

    def forward(self, x):
        seq_len, batch_size, heads, d_k = x.shape
        x = x.permute(1, 2, 3, 0)
        x = x.view(batch_size*heads, d_k, seq_len)
        x = self.conv(x)
        x = x[:, :, :-(self.kernel_size-1)]
        x = x.view(batch_size, heads, d_k, seq_len)
        x = x.permute(3, 0, 1, 2)
        return x


class Attention(pyg_nn.MessagePassing):
    '''
    multihead structure attention
    input:batch of graph data from pyg
    arg:
    embed_dim:int the dimension of the embedding
    num_heads:int the number of the heads
    dropout:float the dropout rate
    bias:bool whether to use bias
    symmetric:bool whether to use symmetric attention
    k:number of k-hop neighbors
    '''

    def __init__(self, embed_dim, num_heads=8, dropout=0.1, bias=False,
                 symmetric=False, k_hop=3, **kwargs):

        super().__init__(node_dim=0, aggr='add')
        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.khop_structure_extractor = KHopStructureExtractor(
            embed_dim, num_layers=k_hop, **kwargs)
        self.attend = nn.Softmax(dim=-1)

        self.symmetric = symmetric
        if symmetric:
            self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

        self.attn_sum = None

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
                x,
                edge_index,
                complete_edge_index,
                subgraph_node_index=None,
                subgraph_edge_index=None,
                subgraph_indicator_index=None,
                subgraph_edge_attr=None,
                edge_attr=None,
                ptr=None,
                return_attn=False):
        """
        Compute attention layer. 

        Args:
        ----------
        x:                          input node features
        edge_index:                 edge index from the graph
        complete_edge_index:        edge index from fully connected graph
        subgraph_node_index:        documents the node index in the k-hop subgraphs
        subgraph_edge_index:        edge index of the extracted subgraphs 
        subgraph_indicator_index:   indices to indicate to which subgraph corresponds to which node
        subgraph_edge_attr:         edge attributes of the extracted k-hop subgraphs
        edge_attr:                  edge attributes
        return_attn:                return attention (default: False)

        """
        # Compute value matrix

        v = self.to_v(x)

        # Compute structure-aware node embeddings
        x_struct = self.khop_structure_extractor(
            x=x,
            edge_index=edge_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_attr=subgraph_edge_attr,
        )

        # Compute query and key matrices
        if self.symmetric:
            qk = self.to_qk(x_struct)
            qk = (qk, qk)
        else:
            qk = self.to_qk(x_struct).chunk(2, dim=-1)

        # Compute complete self-attention
        attn = None

        if complete_edge_index is not None:
            out = self.propagate(complete_edge_index, v=v, qk=qk, edge_attr=None, size=None,
                                 return_attn=return_attn)
            if return_attn:
                attn = self._attn
                self._attn = None
                attn = torch.sparse_coo_tensor(
                    complete_edge_index,
                    attn,
                ).to_dense().transpose(0, 1)

            out = rearrange(out, 'n h d -> n (h d)')
        else:
            out, attn = self.self_attn(qk, v, ptr, return_attn=return_attn)
        return self.out_proj(out), attn

    def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i, return_attn):
        """Self-attention operation compute the dot-product attention """

        qk_i = rearrange(qk_i, 'n (h d) -> n h d', h=self.num_heads)
        qk_j = rearrange(qk_j, 'n (h d) -> n h d', h=self.num_heads)
        v_j = rearrange(v_j, 'n (h d) -> n h d', h=self.num_heads)
        attn = (qk_i * qk_j).sum(-1) * self.scale
        if edge_attr is not None:
            attn = attn + edge_attr
        attn = utils.softmax(attn, index, ptr, size_i)
        if return_attn:
            self._attn = attn
        attn = self.attn_dropout(attn)

        return v_j * attn.unsqueeze(-1)

    def self_attn(self, qk, v, ptr, return_attn=False):
        """ Self attention which can return the attn """

        qk, mask = pad_batch(qk, ptr, return_mask=True)
        k, q = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_heads), qk)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots = dots.masked_fill(
            mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )

        dots = self.attend(dots)
        dots = self.attn_dropout(dots)

        v = pad_batch(v, ptr)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = unpad_batch(out, ptr)

        if return_attn:
            return out, dots
        return out, None
