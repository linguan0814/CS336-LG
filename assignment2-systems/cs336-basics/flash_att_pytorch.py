from sys import dont_write_bytecode
from turtle import forward
from numpy import einsum, float32, tile
from regex import B
from sympy import N
import torch
import math
import einops

class flash_attention_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        device = q.device
        B, N_q, d_model = q.shape
        _, N_k, _       = k.shape
        tile_size = 64
        T_q = math.ceil(N_q / tile_size)
        T_k = math.ceil(N_k / tile_size)
        O = torch.empty((B, N_q, d_model), device=device, dtype=torch.float32)
        L = torch.empty((B, N_q), device=device, dtype=torch.float32)