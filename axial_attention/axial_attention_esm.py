import math
import torch
from torch.autograd import backward
import torch.nn as nn

class RowSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout = 0.0,
        max_tokens_per_msa: int = 2**16,
        ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa
        self.attn_shape = 'hnij'

        self.k_proj, self.v_proj, self.q_proj = nn.Linear(embed_dim, embed_dim), \
            nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim,embed_dim)
        self.out_proj = nn.Linear(embed_dim,embed_dim)
        self.dropout_module = nn.Dropout(self.dropout)
    
    def align_scaling(self,q):
        num_rows = q.size(0)
        # scale / \sqrt(H)
        return self.scaling / math.sqrt(num_rows)
    
    def _batched_forward(
        self,
        x,
        self_attn_mask = None,
        self_attn_padding_mask = None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.shape
        max_rows = max(1, self.max_tokens_per_msa // num_cols)
        attns = 0
        scaling = self.align_scaling(x)
        for start in range(0, num_rows, max_rows):
            attn_weights = self.compute_attention_weights(
                x[start:start+max_rows], 
                scaling, 
                self_attn_mask, 
                self_attn_padding_mask = self_attn_padding_mask[:, start:start+max_rows] 
                if self_attn_padding_mask is not None else None
            )
            attns += attn_weights
        attn_probs = attns.softmax(dim=-1)
        attn_probs = self.dropout_module(attn_probs)

        outputs= []
        for start in range(0, num_rows, max_rows):
            output = self.compute_attention_update(
                x, attn_probs
            )
            outputs.append(output)
        outputs = torch.cat(outputs,0)
        return output, attn_probs
    
    def compute_attention_weights(
        self,
        x,
        scaling: float,
        self_attn_mask = None,
        self_attn_padding_mask = None #B H W
    ):
        num_rows, num_cols, batch_size, embed_dim = x.shape
        q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim) # H W B h d
        k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim) # H W B h d
        q *= scaling
        if self_attn_padding_mask is not None:
            q *= 1 - self_attn_padding_mask.permute(1,2,0).unsqueeze(3).unsqueeze(4).to(q) #expand to (H W B () ())
        
        attn_weights = torch.einsum(f"ribhd, rjbhd->hbij",q,k) # h B W W

        if self_attn_mask is not None:
            raise NotImplementedError
            # Mask Size:[B, H, W]
            # Weight Size: [h, B, W, W]
        
        if self_attn_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                self_attn_padding_mask[:,0].unsqueeze(0).unsqueeze(2), #?
                -10000,
            )
        return attn_weights
    
    def compute_attention_update(
        self,
        x,
        attn_probs
    ):
        num_rows, num_cols, batch_size, embed_dim = x.shape
        v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim) # H W B h d
        context = torch.einsum('hbij,rjbhd->ribhd', attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        return output
    
    def forward(
        self,
        x,
        self_attn_mask = None,
        self_attn_padding_mask = None
    ):
        num_rows, num_cols, batch_size, embed_dim = x.shape
        if (num_cols * num_rows > self.max_tokens_per_msa) and not torch.is_grad_enabled():
            # if H*W too big, split into chunks along row-axis
            return self._batched_forward(x, self_attn_mask, self_attn_padding_mask)
        else:
            scaling = self.align_scaling(x)
            attn_weights = self.compute_attention_weights(x, scaling, self_attn_mask, self_attn_padding_mask)
            attn_probs = attn_weights.softmax(dim=-1)
            attn_probs = self.dropout_module(attn_probs)
            output = self.compute_attention_update(x, attn_probs)
            return output, attn_probs

class ColumnSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout = 0.0,
        max_tokens_per_msa: int = 2**16,
        ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa

        self.k_proj, self.v_proj, self.q_proj = nn.Linear(embed_dim, embed_dim), \
            nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim,embed_dim)
        self.out_proj = nn.Linear(embed_dim,embed_dim)
        self.dropout_module = nn.Dropout(self.dropout)
    
    def _batched_forward(
        self,
        x,
        self_attn_mask = None,
        self_attn_padding_mask = None
    ):
        num_rows, num_cols, batch_size, embed_dim = x.shape
        max_cols = max(1, self.max_tokens_per_msa // num_rows)
        outputs = []
        attns = []
        for start in range(0, num_cols, max_cols):
            output, attn = self(
                x[:,start:start+max_cols],
                self_attn_mask,
                self_attn_padding_mask[:, :, start:start+max_cols] if self_attn_padding_mask else None
            )
            outputs.append(output)
            attns.append(attn)
        outputs = torch.cat(outputs, dim=0)
        attns = torch.cat(attns, dim=0)
        return outputs, attns
    
    def compute_attention_update(
        self,
        x,
        self_attn_mask = None,
        self_attn_padding_mask = None
    ):
        num_rows, num_cols, batch_size, embed_dim = x.shape
        if num_rows == 1:
            attn_probs = torch.ones(
                self.num_heads,
                num_cols,
                batch_size,
                num_rows,
                num_rows,
                device = x.device,
                dtype = x.dtype
            )
            output = self.out_proj(self.v_proj(x))
        else:
            q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim) # H W B h d
            k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim) # H W B h d
            v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim) # H W B h d
            q *= self.scaling
            attn_weights = torch.einsum('icbhd,jcbhd->hcbij',q,k)
            if self_attn_padding_mask is not None:
                #[B,R,C] -> [(),C,B,{},R]
                attn_weights = attn_weights.masked_fill(
                    self_attn_padding_mask.permute(2,0,1).unsqueeze(0).unsqueeze(3),
                    -10000
                )
            attn_probs = attn_weights.softmax(dim=-1)
            attn_probs = self.dropout_module(attn_probs)
            context = torch.einsum('hcbij,jcbhd->icbhd',attn_probs,v)
            context = context.contiguous().view(num_rows,num_cols,batch_size,embed_dim)
            output = self.out_proj(context)
        return output, attn_probs
    
    def forward(
        self,
        x,
        self_attn_mask = None,
        self_attn_padding_mask = None
    ):
        num_rows, num_cols, batch_size, embed_dim = x.shape
        if (num_cols*num_rows > self.max_tokens_per_msa) and not torch.is_grad_enabled():
            return self._batched_forward(x, self_attn_mask, self_attn_padding_mask)
        else:
            return self.compute_attention_update(x, self_attn_mask, self_attn_padding_mask)