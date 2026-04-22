import torch
import torch.nn as nn
from einops import rearrange, einsum
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the paramters on
        dtype: torch.dtype | None = None type of the paramters
        '''
        super().__init__
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.__init__weight()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')
    
    def __init__weight(self):
        std = (2/(self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean = 0, std=std, a=-3*std, b=3*std)


class Embedding(nn.Mudule):
    def __init__(self, num_embeddings, embedding_dim, device= None, dtype=None):
        '''
        num_embeddings: int size of the vocabulary
        embedding_dim : int dimension of the embedding vectors, i.e., dmodel
        device
        dtype
        '''
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embed_weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype == torch.long:
            pass
        else:
            token_ids = token_ids.long()
        return self.embed_weight[token_ids]
    
    def _init_weight(self):
        nn.init.trunc_normal_(self.embed_weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps: float = 1e-5, device=None, dtype=None):
        '''
        d_model: int hidden dimension of the model
        eps: float = 1e-5 epsilon value for numerical stability
        device: torch.device | None = None device to store the parameters on
        dtype: torch.dtype | None = None data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.esp = eps
        self.g_weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self._init_weight()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True)+self.esp)
        out = einsum(x/rms, self.g_weight,'... d, d -> ... d')
        return out.to(dtype=in_dtype)
    
    def _init_weight(self):
        nn.init.trunc_normal_(self.g_weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        '''
        d_model: int dimensionality of the input and output features
        d_ff: int dimensionality of the hidden feed-forward layer
        device: torch.device | None = None device to store the parameters on
        dtype: torch.dtype | None = None data type of the parameters
        '''
        super().__init__()
        if dtype is None or not torch.is_floating_point(torch.empty((), dtype=dtype)):
            dtype = torch.float32
        
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.w1 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype))
        self._init_weight()

    def forward(self, x) -> torch.Tensor:
        '''
        x: torch.Tensor input tensor with shape (..., d_model)

        returns:
        torch.Tensor tensor of shape (..., d_model)

        explanation:
        SwiGLU is a gated feed-forward network. The input x is projected into
        two hidden branches:
        1. one branch is passed through the SiLU activation
        2. the other branch acts as a gate
        these two branches are multiplied elementwise, and then projected back
        to d_model

        mathematically:
        output = W2( SiLU(W1 x) * (W3 x) )
        '''        
        a = einsum(self.w1, x, 'd_ff d_model, ... d_model -> ... d_ff')
        step1 = a*torch.sigmoid(a)
        step2 = step1 * einsum(self.w3, x, 'd_ff d_model, ... d_model -> ... d_ff')
        return einsum(self.w2, step2, 'd_model d_ff, ... d_ff -> ... d_model')
    
    def _init_weight(self):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=1.0, a=-3.0, b=3.0)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=1.0, a=-3.0, b=3.0)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=1.0, a=-3.0, b=3.0)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        theta: float Θ value for the RoPE
        d_k' int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.freq_arrange = 1 / (self.theta**(torch.arange(0, self.d_k, 2).to(dtype=torch.float)/self.d_k))
        self.register_buffer(name='inv_freq', tensor=self.freq_arrange)

        