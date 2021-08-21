import torch 
import deepcopy
import numpy as np 
def expand_dim_to(tensor, length):
	if length==0: return tensor
	return tensor.reshape( *((1,)*length), *tensor.shape)


def MDS(pre_matrix, weights, eigen, verbose, iters, tol):
	device,dtype = pre_matrix.device, pre_matrix.dtype
	#ensure batched MDS ? why
	pre_matrix = expand_dim_as(pre_matrix, length=3-len(pre_matrix.shape))

	batch,N,_ = pre_matrix.shape 
	diag_idxs = np.arange(N)
	his = [torch.tensor([np.inf]*batch, device = device)]

	D = pre_matrix**2
	M = 0.5*(D[:,:1,:]-D[:,:,:1]-D)
	svds = [torch.svd_lowrank(mi) for mi in M]
	u = torch.stack([svd[0] for svd in svds], dim=0)
	s = torch.stack([svd[1] for svd in svds], dim=0)
	v = torch.stack([svd[2] for svd in svds], dim=0)
	best_3d_coords = torch.bmm(u, torch.diag_embed(s).abs().sqrt())[...,:3]

	if weights is None and eigen == True:
		return best_3d_coords.transpose(-1,-2), torch.zeros_like(torch.stack(his,dim=0))
	elif eigen == True:
		if verbose:
			print(',,,')

	if weights is None:
		weights = torch.ones_like(pre_matrix)

	for i in range(iters):
		best_3d_coords = best_3d_coords.contiguous()
		dist_mat = torch.cdist(best_3d_coords, best_3d_coords, p=2).clone()

		stress = (weights * (dist_mat - pre_matrix)**2).sum(dim=(-1,-2)) * 0.5

		dist_mat[dist_mat<=0] += 1e-7
		ratio = weights*(pre_matrix/dist_mat)
		B = -ratio
		B[:,diag_idxs,diag_idxs] += ratio.sum(dim=-1)

        coords = (1./ N * torch.matmul(B, best_3d_coords))
		dis = torch.norm(coords, dim=(-1,-2))

		if verbose >= 2:
            print('it: %d, stress %s' % (i, stress))
        # update metrics if relative improvement above tolerance
        if (his[-1] - stress / dis).mean() <= tol:
            if verbose:
                print('breaking at iteration %d with stress %s' % (i,
                                                                   stress / dis))
            break

        best_3d_coords = coords
        his.append( stress / dis )

    return best_3d_coords.transpose(-1,-2), torch.zeros_like(torch.stack(his,dim=0))