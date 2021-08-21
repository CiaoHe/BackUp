import torch
from torch import nn
from torch import Tensor 
import torch.nn.functional as F

# helper function

def exists(val):
    return val is not None

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# top k filtering

def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class AutogressiveWrapper(nn.Module):
    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    @eval_decorator
    def generate(
        self, 
        start_tokens:Tensor, 
        seq_len, 
        eos_token=None, 
        temperature = 1,
        filter_logits_fn = top_k,
        filter_thres = 0.9,
        **kwargs
    ):
        device = start_tokens.device
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]
        
        b, t = start_tokens.shape

        out = start_tokens

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:] # 每次从最后选取mas_seq_len长度的tokens

            logits = self.net(x, **kwargs)[:, -1, :] # (b, num_tokens)
            
            filtered_logits = top_k(logits=logits, thres=filter_thres) 
            probs = F.softmax(filtered_logits / temperature, dim=-1,)

            sample = torch.multinomial(probs, num_samples=1) # (b, one_token)

            # 不断地将新sample合并到out中
            out = torch.cat((out, sample), dim=-1) # (b, last_tokens + one_token)

            if exists(eos_token):
                is_ens_token = (out == eos_token)

                if is_ens_token.any(dim=-1).all():
                    # mask out everything after the eos tokens (nice trick)
                    shifted_is_eos_tokens = F.pad(is_ens_token, (1,-1)) # shift right
                    mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)
        
        return out
    
    def forward(self, x, **kwargs):
        xi = x[:,:-1]
        xo = x[:,1:]

        out = self.net(xi, **kwargs)
        loss = F.cross_entropy(out.transpose(1,2), xo, ignore_index=self.ignore_index)
        return loss