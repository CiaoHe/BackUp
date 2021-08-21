import torch
from torch import nn, einsum
from torch import Tensor
import torch.nn.functional as F

from .vit import ViT
from .t2t import T2TViT
from .efficient import ViT as EfficientViT

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

# classes

class DistillMixin:
    def forward(self, img, distill_token = None):
        distilling = exists(distill_token)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '{} n d -> b n d', b = b)
        x = torch.cat([cls_tokens, x],dim=1)
        x += self.pos_embedding[:,:(n+1)]

        if distilling:
            distill_tokens = repeat(distill_token, '() n d -> b n d', b=b)
            x = torch.cat([x, distill_tokens],dim=1)
        
        x = self._attend(x)

        if distilling:
            x, distill_tokens = x[:,:-1], x[:,-1] # [b,d]
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:,0]

        x = self.to_latent(x)
        out = self.mlp_head(x)

        if distilling:
            return out, distill_tokens
        else:
            return out

class DistillableViT(ViT, DistillMixin):
    def __init__(self, *args, **kwargs):
        super(DistillableViT,self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = ViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v
    
    def _attend(self, x:Tensor):
        x = self.dropout(x)
        x = self.transformer(x)
        return x

class DistillableT2TViT(T2TViT, DistillMixin):
    def __init__(self, *args, **kwargs):
        super(DistillableT2TViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']
    
    def to_vit(self):
        v = T2TViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v
    
    def _attend(self, x:Tensor):
        x = self.dropout(x)
        x = self.transformer(x)
        return x

class DistillableEfficientViT(EfficientViT, DistillMixin):
    def __init__(self, *args, **kwargs):
        super(DistillableEfficientViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']
    
    def to_vit(self):
        v = EfficientViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v
    
    def _attend(self, x:Tensor):
        x = self.dropout(x)
        x = self.transformer(x)
        return x

# knowledge distillation wrapper

class DistillWrapper(nn.Module):
    def __init__(
        self,
        *,
        teacher:nn.Module,
        student:nn.Module,
        temperature = 1.,
        alpha = 0.5,
        hard = False
    ):
        super().__init__()
        assert (isinstance(student, (DistillableViT, DistillableT2TViT, DistillableEfficientViT))), 'student must be a vision transformer'

        self.teacher = teacher
        self.student = student

        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.hard = hard

        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))

        self.distill_map = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, labels, temperature=None, alpha=None, **kwargs):
        b, *_ = img.shape
        alpha = alpha if exists(alpha) else self.alpha
        T = temperature if exists(temperature) else self.temperature

        # first get teacher's logits
        with torch.no_grad():
            teacher_logits = self.teacher(img)
        
        student_logits, distill_tokens = self.student(img, distill_token = self.distillation_token, **kwargs)
        distill_logits = self.distill_map(distill_tokens)

        ce_loss = F.cross_entropy(student_logits, labels)

        if not self.hard:
            # if use soft distillation
            distill_loss = F.kl_div(
                F.log_softmax(distill_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1).detach(),
                reduction='batchmean',
            )
            distill_loss *= T ** 2
        
        else:
            # use hard distillation
            distill_loss = F.cross_entropy(student_logits, teacher_logits.argmax(dim=-1))
        
        return (1-self.alpha) * ce_loss + self.alpha * distill_loss