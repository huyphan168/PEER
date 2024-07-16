import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import math
from einops import einsum
from tqdm import tqdm 
from einops.layers.torch import Rearrange

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma

class ProductKeyMemory(nn.Module):
    def __init__(self, dim, num_keys):
        super().__init__()
        self.dim = dim
        self.num_keys = num_keys
        self.keys = nn.Parameter(torch.randn(num_keys, dim // 2))
        
    def forward(self, query):
        query = query.view(query.shape[0], 2, -1)
        dots = torch.einsum('bkd,nd->bkn', query, self.keys)
        return dots.view(query.shape[0], -1)

class PEER(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads=8,
        num_experts=1_000_000,
        num_experts_per_head=16,
        activation=nn.GELU,
        dim_key=None,
        product_key_topk=None,
        separate_embed_per_head=False,
        pre_rmsnorm=False,
        dropout=0.
    ):
        super().__init__()

        self.norm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        self.heads = heads
        self.separate_embed_per_head = separate_embed_per_head
        self.num_experts = num_experts

        num_expert_sets = heads if separate_embed_per_head else 1

        self.weight_down_embed = nn.Embedding(num_experts * num_expert_sets, dim)
        self.weight_up_embed = nn.Embedding(num_experts * num_expert_sets, dim)

        self.activation = activation()

        assert (num_experts ** 0.5).is_integer(), '`num_experts` needs to be a square'
        assert (dim % 2) == 0, 'feature dimension should be divisible by 2'

        dim_key = default(dim_key, dim // 2)
        self.num_keys = int(num_experts ** 0.5)

        self.to_queries = nn.Sequential(
            nn.Linear(dim, dim_key * heads * 2, bias=False),
            Rearrange('b n (p h d) -> p b n h d', p=2, h=heads)
        )

        self.product_key_topk = default(product_key_topk, num_experts_per_head)
        self.num_experts_per_head = num_experts_per_head

        self.keys = nn.Parameter(torch.randn(heads, self.num_keys, 2, dim_key))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)

        queries = self.to_queries(x)

        sim = einsum(queries, self.keys, 'p b n h d, h k p d -> p b n h k')

        (scores_x, scores_y), (indices_x, indices_y) = [s.topk(self.product_key_topk, dim=-1) for s in sim]

        all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
        all_indices = indices_x.unsqueeze(-1) * self.num_keys + indices_y.unsqueeze(-2)

        all_scores = all_scores.view(*all_scores.shape[:-2], -1)
        all_indices = all_indices.view(*all_indices.shape[:-2], -1)

        scores, pk_indices = all_scores.topk(self.num_experts_per_head, dim=-1)
        indices = all_indices.gather(-1, pk_indices)

        if self.separate_embed_per_head:
            head_expert_offsets = torch.arange(self.heads, device=x.device) * self.num_experts
            indices = indices + head_expert_offsets.view(1, 1, -1, 1)

        weights_down = self.weight_down_embed(pk_indices)
        weights_up = self.weight_up_embed(pk_indices)

        x = einsum(x, weights_down, 'b n d, b n h k d -> b n h k')

        x = self.activation(x)
        x = self.dropout(x)

        x = x * F.softmax(scores, dim=-1)

        x = einsum(x, weights_up, 'b n h k, b n h k d -> b n d')

        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_experts, num_experts_per_head, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.peer1 = PEER(dim, heads=num_heads, num_experts=num_experts, num_experts_per_head=num_experts_per_head)
        self.peer2 = PEER(dim, heads=num_heads, num_experts=num_experts, num_experts_per_head=num_experts_per_head)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        peer_output1 = self.peer1(x)
        peer_output2 = self.peer2(F.gelu(peer_output1))
        x = x + self.dropout(peer_output2)
        x = self.norm2(x)
        
        return x

class PEERLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, num_experts, top_k):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(512, dim)  
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads, num_experts, top_k) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, x):
        b, s = x.shape
        positions = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        
        x = self.token_embedding(x) + self.position_embedding(positions)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        return logits