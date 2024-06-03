import torch
import torch.nn as nn
import numpy as np
import json
import random

class TripleGrainFixedEntropyRouter(nn.Module):
    def __init__(self, coarse_grain_ratio, medium_grain_ratio):
        super().__init__()

        self.coarse_grain_ratio = coarse_grain_ratio
        self.medium_grain_ratio = medium_grain_ratio
        self.fine_grain_ratio = 1 - coarse_grain_ratio - medium_grain_ratio
    
    def forward(self, x_entropy_p16, x_entropy_p8):
        mode = 0
        if (self.fine_grain_ratio == 0) + (self.medium_grain_ratio == 0) + (self.coarse_grain_ratio == 0) == 0:
            mode = 0
            x_entropy_p16_flatten = x_entropy_p16.flatten() # torch.Size([1, 16, 24]) 
            x_entropy_p16_sorted, _ = torch.sort(x_entropy_p16_flatten, descending=False)
            k_coarse = round(x_entropy_p16_flatten.shape[-1]*self.coarse_grain_ratio)
            coarse_entropy_threshold = x_entropy_p16_sorted[k_coarse - 1 if k_coarse!=0 else k_coarse]
            gate_coarse = torch.where(x_entropy_p16 < coarse_entropy_threshold, 1, 0).bool().int() # torch.Size([1, 16, 24])

            x_entropy_p8_flatten = (x_entropy_p8*(1-gate_coarse.repeat_interleave(2,dim=-1).repeat_interleave(2,dim=-2))).flatten()
            x_entropy_p8_sorted, _ = torch.sort(x_entropy_p8_flatten, descending=False)

            k_medium_4coarse = round(4*x_entropy_p16_flatten.shape[-1]*self.coarse_grain_ratio + x_entropy_p8_flatten.shape[-1]*self.medium_grain_ratio)
            medium_entropy_threshold = x_entropy_p8_sorted[(k_medium_4coarse - 1) if k_medium_4coarse!=0 else k_medium_4coarse]
            gate_medium = torch.where(x_entropy_p8 < medium_entropy_threshold, 1, 0)*torch.where((1-gate_coarse.repeat_interleave(2,dim=-1).repeat_interleave(2,dim=-2)).bool(), 1, 0)
            gate_medium = gate_medium.bool().int()

            gate_fine = 1 - gate_coarse.repeat_interleave(4,dim=-1).repeat_interleave(4,dim=-2) - gate_medium.repeat_interleave(2,dim=-1).repeat_interleave(2,dim=-2)

        elif (self.fine_grain_ratio == 0) + (self.medium_grain_ratio == 0) + (self.coarse_grain_ratio == 0) == 1:
            # One of the ratios is zero
            if self.coarse_grain_ratio == 0:
                mode = 1
                x_entropy_p8_flatten = x_entropy_p8.flatten()
                x_entropy_p8_sorted, _ = torch.sort(x_entropy_p8_flatten, descending=False)
                k_medium = round(x_entropy_p8_flatten.shape[-1]*self.medium_grain_ratio)
                medium_entropy_threshold = x_entropy_p8_sorted[k_medium - 1 if k_medium!=0 else k_medium]
                gate_medium = torch.where(x_entropy_p8 < medium_entropy_threshold, 1, 0).bool().int()
                gate_medium = gate_medium.bool().int()

                gate_fine = 1 - gate_medium.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
                gate_coarse = torch.zeros([gate_medium.shape[0], gate_medium.shape[1]//2, gate_medium.shape[2]//2]).bool().int().to(x_entropy_p16.device)

            elif self.medium_grain_ratio == 0:
                mode = 2
                x_entropy_p16_flatten = x_entropy_p16.flatten() 
                x_entropy_p16_sorted, _ = torch.sort(x_entropy_p16_flatten, descending=False)
                k_coarse = round(x_entropy_p16_flatten.shape[-1]*self.coarse_grain_ratio)
                coarse_entropy_threshold = x_entropy_p16_sorted[k_coarse - 1 if k_coarse!=0 else k_coarse]
                gate_coarse = torch.where(x_entropy_p16 < coarse_entropy_threshold, 1, 0).bool().int() 

                gate_fine = 1 - gate_coarse.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
                gate_medium = torch.zeros([gate_coarse.shape[0], gate_coarse.shape[1]*2, gate_coarse.shape[2]*2]).bool().int().to(x_entropy_p16.device)

            else: # fine_grain_ratio == 0
                mode = 3
                x_entropy_p16_flatten = x_entropy_p16.flatten()
                x_entropy_p16_sorted, _ = torch.sort(x_entropy_p16_flatten, descending=False)
                k_coarse = round(x_entropy_p16_flatten.shape[-1]*self.coarse_grain_ratio)
                coarse_entropy_threshold = x_entropy_p16_sorted[k_coarse - 1 if k_coarse!=0 else k_coarse]
                gate_coarse = torch.where(x_entropy_p16 < coarse_entropy_threshold, 1, 0).bool().int()

                gate_medium = 1 - gate_coarse.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
                gate_fine = torch.zeros([gate_coarse.shape[0], gate_coarse.shape[1]*4, gate_coarse.shape[2]*4]).bool().int().to(x_entropy_p16.device)

        else:   # (fine_grain_ratio == 0) + (medium_grain_ratio == 0) + (coarse_grain_ratio == 0) == 2
            # Two of the ratios are zero
            if self.coarse_grain_ratio != 0:
                mode = 4
                gate_coarse = torch.ones_like(x_entropy_p16).bool().int().to(x_entropy_p16.device)
                gate_medium = torch.zeros_like(x_entropy_p8).bool().int().to(x_entropy_p16.device)
                gate_fine = gate_medium.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)

            elif self.medium_grain_ratio != 0:
                mode = 5
                gate_medium = torch.ones_like(x_entropy_p8).bool().int().to(x_entropy_p16.device)
                gate_coarse = torch.zeros_like(x_entropy_p16).bool().int().to(x_entropy_p16.device)
                gate_fine = gate_coarse.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)

            else: # fine_grain_ratio != 0:
                mode = 6
                gate_fine = torch.ones([x_entropy_p8.shape[0], x_entropy_p8.shape[1]*2, x_entropy_p8.shape[2]*2]).bool().int().to(x_entropy_p16.device)
                gate_medium = torch.zeros_like(x_entropy_p8).bool().int().to(x_entropy_p16.device).to(x_entropy_p16.device)
                gate_coarse = torch.zeros_like(x_entropy_p16).bool().int().to(x_entropy_p16.device).to(x_entropy_p16.device)
        
        mask = [gate_coarse, gate_medium, gate_fine]
        gate = torch.cat([gate_coarse.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2).unsqueeze(-1), gate_medium.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2).unsqueeze(-1), gate_fine.unsqueeze(-1)], dim=-1)

        return mask, gate, [self.coarse_grain_ratio, self.medium_grain_ratio, self.fine_grain_ratio], mode
