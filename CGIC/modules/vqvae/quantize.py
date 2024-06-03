import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torch import einsum

class VectorQuantize2(nn.Module):
    def __init__(self,
                n_e,
                e_dim,
                beta, 
                remap=None, 
                unknown_index="random",
                sane_index_shape=False, 
                legacy=True
                ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        self.embedding_counter = nn.ParameterDict({str(i): nn.Parameter(torch.zeros(1)) for i in range(n_e)}).requires_grad_(False).cuda()

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)
    
    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)
    
    def forward(self, z):
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flatten = z.view(-1, self.e_dim)

        d = torch.sum(z_flatten ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flatten, rearrange(self.embedding.weight, 'n d -> d n'))
        
        # multi-grain z_indices calculation
        z_indices = torch.argmin(d, dim=1)

        for index in z_indices:
            self.embedding_counter[str(index.item())] += 1  # update

        z_q = self.embedding(z_indices).view(z.shape)

        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        return z_q, loss, z_indices

    # @torch.no_grad()
    # def get_soft_codes(self, x, temp=1.0, stochastic=False):
    #     distances = self.embedding.compute_distances(x)
    #     soft_code = F.softmax(-distances / temp, dim=-1)

    #     if stochastic:
    #         soft_code_flat = soft_code.reshape(-1, soft_code.shape[-1])
    #         code = torch.multinomial(soft_code_flat, 1)
    #         code = code.reshape(*soft_code.shape[:-1])
    #     else:
    #         code = distances.argmin(dim=-1)

    #     return soft_code, code
    
    # def get_codebook_entry(self, indices, *kwargs):
    #     # get quantized latent vectors
    #     indices_coarse = indices[0] - 1
    #     indices_fine = indices[1] - 1
        
    #     coarse_fill = (torch.zeros_like(indices_coarse)).to(indices_coarse.device).to(indices_coarse.dtype)
    #     fine_fill = coarse_fill.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)

    #     indices_coarse_embed = torch.where(indices_coarse==-1, coarse_fill, indices_coarse)
    #     indices_fine_embed = torch.where(indices_fine==-1, fine_fill, indices_fine)

    #     z_q_coarse = self.embedding(indices_coarse_embed)
    #     z_q_fine = self.embedding(indices_fine_embed)

    #     z_q_coarse = z_q_coarse.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-3)

    #     fine_mask = torch.where(indices_fine==-1, fine_fill, 1-fine_fill).unsqueeze(3).repeat_interleave(256, dim=-1)
    #     z_q = torch.where(fine_mask==0, z_q_coarse, z_q_fine)

    #     return z_q