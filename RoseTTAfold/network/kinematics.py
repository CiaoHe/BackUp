import numpy as np 
import torch
from einops import rearrange,reduce,repeat

PARAMS = {
    "DMIN"    : 2.0, # 最小距离
    "DMAX"    : 20.0,# 最大距离
    "DBINS"   : 36,  # 划分多少个bins
    "ABINS"   : 36,
}

def get_pair_dist(a,b):
    """calculate pair distances between two sets of points
    
    Parameters
    ----------
    a,b : pytorch tensors of shape [batch,nres,3]
          store Cartesian coordinates of two sets of atoms
    Returns
    -------
    dist : pytorch tensor of shape [batch,nres,nres]
           stores paitwise distances between atoms in a and b
    """

    dist = torch.cdist(a, b, p=2)
    return dist

def get_ang(a, b, c):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
    from Cartesian coordinates of three sets of atoms a,b,c 

    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """

    '''
    v = a - b (vec: b->a)
    w = c - b (vec: b->c)
    v = v/||v||
    w = w/||w||
    cos = v \dot w
    '''

    v = a - b
    w = c - b 
    v /= torch.norm(v, dim=-1, keepdim=True) # normalize last dim
    w /= torch.norm(w, dim=-1, keepdim=True)
    vw = torch.sum(v * w, dim=-1)

    return torch.acos(vw)

def get_dih(a, b, c, d):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors of shape [batch,nres,3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor of shape [batch,nres]
          stores resulting dihedrals
    """
    b0 = a - b
    b1 = c - b 
    b2 = d - c

    b1 /= torch.norm(b1, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0*b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2*b1, dim=-1, keepdim=True) * b1

    x = torch.sum(v*2, dim=-1)
    y = torch.sum(torch.cross(b1,v,dim=-1)*2, dim=-1)

    return torch.atan2(y,x)

# ============================================================
def xyz_to_c6d(xyz, params=PARAMS):
    """convert cartesian coordinates into 2d distance 
    and orientation maps
    
    Parameters
    ----------
    xyz : pytorch tensor of shape [batch,nres,3,3]
          stores Cartesian coordinates of backbone N,Ca,C atoms
    Returns
    -------
    c6d : pytorch tensor of shape [batch,nres,nres,4]
          stores stacked dist,omega,theta,phi 2D maps 
    """
    
    batch = xyz.shape[0]
    nres = xyz.shape[1]

    # three achor atoms
    N = xyz[:, :, 0] # (b, nres, 3)
    Ca= xyz[:, :, 1] # (b, nres, 3)
    C = xyz[:, :, 2] # (b, nres, 3)

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca  

    # 6d coordinates order: (dist, omega, theta, phi)
    c6d = torch.zeros([batch, nres, nres, 4], dtype=xyz.dtype, device=xyz.device)

    dist = get_pair_dist(Cb, Cb) # (b, nres, nres)
    dist[torch.isnan(dist)] = 999.9
    c6d[..., 0] = dist + 999.9*torch.eye(nres, device=xyz.device)[None, ...] # 本aa之间距离设置为无穷
    b,i,j = torch.where(c6d[...,0] < params['DMAX']) # 各维度的index

    # torsion(omega) = torsion(Ca_i, C_i, N_j, Ca_j)
    c6d[b,i,j,torch.full_like(b,1)] = get_dih(Ca[b,i], Cb[b,i], Cb[b,j], Ca[b,j])
    c6d[b,i,j,torch.full_like(b,2)] = get_dih(N[b,i], Ca[b,i], Cb[b,i], Cb[b,j])
    c6d[b,i,j,torch.full_like(b,3)] = get_ang(Ca[b,i], Cb[b,i], Cb[b,j])

    # fix long-range distance
    c6d[...,0][c6d[...,0]>=params['DMAX']] = 999.9

    mask = torch.zeros((batch, nres,nres), dtype=xyz.dtype, device=xyz.device)
    mask[b,i,j] = 1.0
    return c6d, mask

def xyz_to_t2d(xyz_t, t0d, params=PARAMS):
    """convert template cartesian coordinates into 2d distance 
    and orientation maps
    
    Parameters
    ----------
    xyz_t : pytorch tensor of shape [batch,templ,nres,3,3]
            stores Cartesian coordinates of template backbone N,Ca,C atoms
    t0d:  0-D template features (HHprob, seqID, similarity) [batch, templ, 3]

    Returns
    -------
    t2d : pytorch tensor of shape [batch,nres,nres,1+6+3]
          stores stacked dist,omega,theta,phi 2D maps 
    """

    B, T, L = xyz_t.shape[:3]
    c6d, mask = xyz_to_c6d(xyz_t.view(B*T, L, 3, 3), params=PARAMS)
    c6d = c6d.view(B, T, L, L, 4)
    mask = mask.view(B, T, L, L, 1)
    #
    dist = c6d[...,:1] * mask / params['DMAX'] # from 0 to 1 # (B,T,L,L,1)
    dist = torch.clamp(dist, 0., 1.)
    orien = torch.cat([c6d[...,1:].sin(), c6d[...,1:].cos()], dim=-1) # (B,T,L,L,6)
    t0d = rearrange(t0d, 'B T r -> B T L L r', L=L)
    #
    t2d = torch.cat((dist, orien, t0d), dim=-1)
    t2d[torch.isnan(t2d)] = 0.0
    return t2d

