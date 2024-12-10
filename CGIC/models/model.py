import json
import os
import pathlib
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import partial

from CGIC.util import instantiate_from_config
from CGIC.modules.vqvae.vqvae_blocks import Encoder
from CGIC.modules.vqvae.decoder import Decoder
from CGIC.modules.vqvae.quantize import VectorQuantize2 as VectorQuantizer
from CGIC.models.ema import LitEma
from CGIC.modules.util import disabled_train
from CGIC.modules.draw import draw_triple_grain_256res_color, draw_triple_grain_256res

from einops import rearrange
import numpy as np

class CGIC(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 learning_rate=None,
                 lossconfig=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ema_decay=None,
                 image_size=256,
                 entropy_patch_size=(8,16)
                 ):
        super().__init__()

        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(zq_ch=embed_dim, **ddconfig)
        if lossconfig is not None:
            self.loss = instantiate_from_config(lossconfig)
        if learning_rate is not None:
            self.learning_rate = learning_rate
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.entropy_calculation_p8 = Entropy(entropy_patch_size[0])
        self.entropy_calculation_p8 = self.entropy_calculation_p8.eval()
        self.entropy_calculation_p8.train = disabled_train

        self.entropy_calculation_p16 = Entropy(entropy_patch_size[1])
        self.entropy_calculation_p16 = self.entropy_calculation_p16.eval()
        self.entropy_calculation_p16.train = disabled_train

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ema_decay is not None:
            self.use_ema = True
            print('use_ema = True')
            self.ema_encoder = LitEma(self.encoder, ema_decay)
            self.ema_decoder = LitEma(self.decoder, ema_decay)
            self.ema_quantize = LitEma(self.quantize, ema_decay) 
            self.ema_quant_conv = LitEma(self.quant_conv, ema_decay) 
            self.ema_post_quant_conv = LitEma(self.post_quant_conv, ema_decay) 
            
        else:
            self.use_ema = False
            
    def init_from_ckpt(self, path, ignore_keys=list()):
        ckpt = torch.load(path, map_location="cpu")
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        x_entropy_p8 = self.entropy_calculation_p8(x)
        x_entropy_p16 = self.entropy_calculation_p16(x)

        h_dict = self.encoder(x, x_entropy_p16, x_entropy_p8)
        h = h_dict["h"]
        grain_indices = h_dict["indices"]
        grain_mask = h_dict["mask"]
        fine_ratio = h_dict["fine_ratio"]
        compression_mode = h_dict["compression_mode"]

        h = self.quant_conv(h)
        quant, emb_loss, ind = self.quantize(h)
        return quant, emb_loss, grain_indices, grain_mask, ind, fine_ratio, compression_mode

    def decode(self, quant, mask):
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant, mask)
        return dec

    def decode_code(self, code_b):
        batch_size = code_b.shape[0]
        quant = self.quantize.embedding(code_b.flatten())
        grid_size = int((quant.shape[0] // batch_size)**0.5)
        quant = quant.view((1, 32, 32, 4))
        quant = rearrange(quant, 'b h w c -> b c h w').contiguous()
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant)
        return dec

    def forward(self, input):
        quant, diff, grain_indices, grain_mask, _, _, _ = self.encode(input)
        if self.trainer.state.stage == 'validation':
            grain_mask[0] = grain_mask[0].squeeze()
            grain_mask[1] = grain_mask[1].squeeze()
            grain_mask[2] = grain_mask[2].squeeze()

        dec = self.decode(quant, grain_mask)
        return dec, diff, grain_indices

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.size(1) != 3:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_encoder(self.encoder)
            self.ema_decoder(self.decoder)
            self.ema_quantize(self.quantize)
            self.ema_quant_conv(self.quant_conv)
            self.ema_post_quant_conv(self.post_quant_conv)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch
        xrec, qloss, grain_indices = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, 
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, 
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = batch
        xrec, qloss, grain_indices = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["val/rec_loss"]
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))

        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        
        return [opt_ae, opt_disc], []

    def compress(self, input, path, h_indices, h_mask, save_img):
        assert len(input.shape)==4
        bpp = 0.0

        # compress
        quant, diff, grain_indices, grain_mask, ind, _, mode = self.encode(input)

        if save_img:
            partition_map = draw_triple_grain_256res(images=input.clone(), indices=grain_indices)
        else:
            partition_map = None
        ind = ind.view(-1, quant.shape[-2], quant.shape[-1])

        ind_coarse = ind[:,::4,::4][grain_mask[0][0] == 1]
        ind_medium = ind[:,::2,::2][grain_mask[1][0] == 1]
        ind_fine = ind[grain_mask[2][0] == 1]

        num_pixels = input.shape[2]*input.shape[3]

        if mode == 0:
            indices_coarse = h_indices.compress(ind_coarse, os.path.join(path, 'indices_coarse.bin')) 
            indices_medium = h_indices.compress(ind_medium, os.path.join(path, 'indices_medium.bin')) 
            indices_fine = h_indices.compress(ind_fine, os.path.join(path, 'indices_fine.bin'))

            mask_coarse = h_mask.compress(grain_mask[0].flatten(), os.path.join(path, 'mask_coarse.bin'))
            mask_medium = h_mask.compress(grain_mask[1].flatten(), os.path.join(path, 'mask_medium.bin'))   

            bpp = (os.path.getsize(indices_coarse) + os.path.getsize(indices_medium) + os.path.getsize(indices_fine) + os.path.getsize(mask_coarse) + os.path.getsize(mask_medium)) * 8 / num_pixels
        elif mode == 1:
            indices_medium = h_indices.compress(ind_medium, os.path.join(path, 'indices_medium.bin')) 
            indices_fine = h_indices.compress(ind_fine, os.path.join(path, 'indices_fine.bin'))
            mask_medium = h_mask.compress(grain_mask[1].flatten(), os.path.join(path, 'mask_medium.bin')) 

            bpp = (os.path.getsize(indices_medium) + os.path.getsize(indices_fine) + os.path.getsize(mask_medium)) * 8 / num_pixels
        elif mode == 2:
            indices_coarse = h_indices.compress(ind_coarse, os.path.join(path, 'indices_coarse.bin')) 
            indices_fine = h_indices.compress(ind_fine, os.path.join(path, 'indices_fine.bin'))
            mask_coarse = h_mask.compress(grain_mask[0].flatten(), os.path.join(path, 'mask_coarse.bin'))

            bpp = (os.path.getsize(indices_coarse) + os.path.getsize(indices_fine) + os.path.getsize(mask_coarse)) * 8 / num_pixels
        elif mode == 3:
            indices_coarse = h_indices.compress(ind_coarse, os.path.join(path, 'indices_coarse.bin')) 
            indices_medium = h_indices.compress(ind_medium, os.path.join(path, 'indices_medium.bin')) 
            mask_coarse = h_mask.compress(grain_mask[0].flatten(), os.path.join(path, 'mask_coarse.bin'))

            bpp = (os.path.getsize(indices_coarse) + os.path.getsize(indices_medium) + os.path.getsize(mask_coarse)) * 8 / num_pixels
        elif mode == 4:
            indices_coarse = h_indices.compress(ind_coarse, os.path.join(path, 'indices_coarse.bin')) 
            bpp = os.path.getsize(indices_coarse) * 8 / num_pixels
        elif mode == 5:
            indices_medium = h_indices.compress(ind_medium, os.path.join(path, 'indices_medium.bin')) 
            bpp = os.path.getsize(indices_medium) * 8 / num_pixels
        else: #mode=6
            indices_fine = h_indices.compress(ind_fine, os.path.join(path, 'indices_fine.bin'))
            bpp = os.path.getsize(indices_fine) * 8 / num_pixels
        
        # print(f'indices_coarse: {os.path.getsize(indices_coarse) * 8 / num_pixels} bpp')
        # print(f'indices_medium: {os.path.getsize(indices_medium) * 8 / num_pixels} bpp')
        # print(f'indices_fine: {os.path.getsize(indices_fine) * 8 / num_pixels} bpp')
        # print(f'mask_coarse: {os.path.getsize(mask_coarse) * 8 / num_pixels} bpp')
        # print(f'mask_medium: {os.path.getsize(mask_medium) * 8 / num_pixels} bpp')
        
        # decompress
        if mode==0:
            ind_coarse_decompress = h_indices.decompress_string(os.path.join(path, 'indices_coarse.bin'))
            ind_medium_decompress = h_indices.decompress_string(os.path.join(path, 'indices_medium.bin'))
            ind_fine_decompress = h_indices.decompress_string(os.path.join(path, 'indices_fine.bin'))

            mask_coarse_decompress = h_mask.decompress_string(os.path.join(path, 'mask_coarse.bin'))
            mask_medium_decompress = h_mask.decompress_string(os.path.join(path, 'mask_medium.bin'))

            # For convenience, the name of the variable has not been changed, and is actually represented here as mask
            ind_coarse = torch.tensor(mask_coarse_decompress).view(1, ind.shape[-2]//4, ind.shape[-1]//4).to(ind.device)
            ind_medium = torch.tensor(mask_medium_decompress).view(1, ind.shape[-2]//2, ind.shape[-1]//2).to(ind.device)
            ind_fine = (1 - ind_medium.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2) - ind_coarse.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)).to(ind.device)

            grain_mask_decompress = [ind_coarse.clone(), ind_medium.clone(), ind_fine.clone()]

            if ind_coarse_decompress is None:
                ind_coarse = torch.zeros([ind_medium.shape[0], ind_medium.shape[1]//2, ind_medium.shape[2]//2]).to(ind.device).int()
            else:
                ind_coarse[ind_coarse==1] = torch.tensor(ind_coarse_decompress).to(ind.device)
            if ind_medium_decompress is None:
                ind_medium = torch.zeros([ind_coarse.shape[0], ind_coarse.shape[1]*2, ind_coarse.shape[2]*2]).to(ind.device).int()
            else:
                ind_medium[ind_medium==1] = torch.tensor(ind_medium_decompress).to(ind.device)
            ind_fine[ind_fine==1] = torch.tensor(ind_fine_decompress).to(ind.device)
            ind_decompress = ind_fine + ind_medium.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2) + ind_coarse.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)
            # print(ind_decompress)

        elif mode==1:
            ind_medium_decompress = h_indices.decompress_string(os.path.join(path, 'indices_medium.bin'))
            ind_fine_decompress = h_indices.decompress_string(os.path.join(path, 'indices_fine.bin'))
            mask_medium_decompress = h_mask.decompress_string(os.path.join(path, 'mask_medium.bin'))

            ind_medium = torch.tensor(mask_medium_decompress).view(1, ind.shape[-2]//2, ind.shape[-1]//2).to(ind.device)
            ind_fine = (1 - ind_medium.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)).to(ind.device)
            ind_coarse = torch.zeros([ind_medium.shape[0], ind_medium.shape[1]//2, ind_medium.shape[2]//2]).to(ind.device)

            grain_mask_decompress = [ind_coarse.clone(), ind_medium.clone(), ind_fine.clone()]
            
            if ind_medium_decompress is None:
                ind_medium = torch.zeros([ind_coarse.shape[0], ind_coarse.shape[1]*2, ind_coarse.shape[2]*2]).to(ind.device).int()
            else:
                ind_medium[ind_medium==1] = torch.tensor(ind_medium_decompress).to(ind.device)
            ind_fine[ind_fine==1] = torch.tensor(ind_fine_decompress).to(ind.device)
            ind_decompress = ind_fine + ind_medium.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)

        elif mode==2:
            ind_coarse_decompress = h_indices.decompress_string(os.path.join(path, 'indices_coarse.bin'))
            ind_fine_decompress = h_indices.decompress_string(os.path.join(path, 'indices_fine.bin'))
            mask_coarse_decompress = h_mask.decompress_string(os.path.join(path, 'mask_coarse.bin'))

            ind_coarse = torch.tensor(mask_coarse_decompress).view(1, ind.shape[-2]//4, ind.shape[-1]//4).to(ind.device)
            ind_fine = (1 - ind_coarse.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)).to(ind.device)
            ind_medium = torch.zeros([ind_coarse.shape[0], ind_coarse.shape[1]*2, ind_coarse.shape[2]*2]).to(ind.device)

            grain_mask_decompress = [ind_coarse.clone(), ind_medium.clone(), ind_fine.clone()]

            if ind_coarse_decompress is None:
                    ind_coarse = torch.zeros([ind_medium.shape[0], ind_medium.shape[1]//2, ind_medium.shape[2]//2]).to(ind.device).int()
            else:
                ind_coarse[ind_coarse==1] = torch.tensor(ind_coarse_decompress).to(ind.device)
            ind_fine[ind_fine==1] = torch.tensor(ind_fine_decompress).to(ind.device)

            ind_decompress = ind_fine + ind_coarse.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)

        elif mode==3:
            ind_coarse_decompress = h_indices.decompress_string(os.path.join(path, 'indices_coarse.bin'))
            ind_medium_decompress = h_indices.decompress_string(os.path.join(path, 'indices_medium.bin'))
            mask_coarse_decompress = h_mask.decompress_string(os.path.join(path, 'mask_coarse.bin'))

            ind_coarse = torch.tensor(mask_coarse_decompress).view(1, ind.shape[-2]//4, ind.shape[-1]//4).to(ind.device)
            ind_medium = (1 - ind_coarse.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)).to(ind.device)
            ind_fine = torch.zeros([ind_coarse.shape[0], ind_coarse.shape[1]*4, ind_coarse.shape[2]*4]).to(ind.device)

            grain_mask_decompress = [ind_coarse.clone(), ind_medium.clone(), ind_fine.clone()]

            if ind_coarse_decompress is None:
                ind_coarse = torch.zeros([ind_medium.shape[0], ind_medium.shape[1]//2, ind_medium.shape[2]//2]).to(ind.device).int()
            else:
                ind_coarse[ind_coarse==1] = torch.tensor(ind_coarse_decompress).to(ind.device)
            if ind_medium_decompress is None:
                ind_medium = torch.zeros([ind_coarse.shape[0], ind_coarse.shape[1]*2, ind_coarse.shape[2]*2]).to(ind.device).int()
            else:
                ind_medium[ind_medium==1] = torch.tensor(ind_medium_decompress).to(ind.device)

            ind_decompress = ind_coarse.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2) + ind_medium.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
            
        elif mode==4:
            ind_coarse_decompress = h_indices.decompress_string(os.path.join(path, 'indices_coarse.bin'))
            ind_coarse_decompress = torch.tensor(ind_coarse_decompress).view(1, ind.shape[-2]//4, ind.shape[-1]//4)
            ind_coarse_decompress = ind_coarse_decompress.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2).int().to(ind.device)
            
            mask_coarse = torch.ones([1, ind.shape[-2]//4, ind.shape[-1]//4]).int().to(ind.device)
            mask_medium = torch.zeros([mask_coarse.shape[0], mask_coarse.shape[1]*2, mask_coarse.shape[2]*2]).int().to(ind.device)
            mask_fine = torch.zeros([mask_coarse.shape[0], mask_coarse.shape[1]*4, mask_coarse.shape[2]*4]).int().to(ind.device)

            grain_mask_decompress = [mask_coarse, mask_medium, mask_fine]

            ind_decompress = ind_coarse_decompress

        elif mode==5:
            ind_medium_decompress = h_indices.decompress_string(os.path.join(path, 'indices_medium.bin'))
            ind_medium_decompress = torch.tensor(ind_medium_decompress).view(1, ind.shape[-2]//2, ind.shape[-1]//2)
            ind_medium_decompress = ind_medium_decompress.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2).int().to(ind.device)

            mask_medium = torch.ones([1, ind.shape[-2]//2, ind.shape[-1]//2]).int().to(ind.device)
            mask_coarse = torch.zeros([mask_medium.shape[0], mask_medium.shape[1]//2, mask_medium.shape[2]//2]).int().to(ind.device)
            mask_fine = torch.zeros([mask_medium.shape[0], mask_medium.shape[1]*2, mask_medium.shape[2]*2]).int().to(ind.device)

            grain_mask_decompress = [mask_coarse, mask_medium, mask_fine]

            ind_decompress = ind_medium_decompress
        else:
            ind_fine_decompress = h_indices.decompress_string(os.path.join(path, 'indices_fine.bin'))

            mask_fine = torch.ones([1, ind.shape[-2], ind.shape[-1]]).int().to(ind.device)
            mask_coarse = torch.zeros([mask_fine.shape[0], mask_fine.shape[1]//4, mask_fine.shape[2]//4]).int().to(ind.device)
            mask_medium = torch.zeros([mask_fine.shape[0], mask_fine.shape[1]//2, mask_fine.shape[2]//2]).int().to(ind.device)

            grain_mask_decompress = [mask_coarse, mask_medium, mask_fine]

            ind_decompress = torch.tensor(ind_fine_decompress).view(1, ind.shape[-2], ind.shape[-1]).int().to(ind.device)

        quant_decompress = self.quantize.embedding(ind_decompress.flatten()).view(ind_decompress.shape[0], ind_decompress.shape[1], ind_decompress.shape[2], -1)
        quant_decompress = rearrange(quant_decompress, 'b h w c -> b c h w')

        if not self.training:
            grain_mask_decompress[0] = grain_mask_decompress[0].unsqueeze(0)
            grain_mask_decompress[1] = grain_mask_decompress[1].unsqueeze(0)
            grain_mask_decompress[2] = grain_mask_decompress[2].unsqueeze(0)

        dec = self.decode(quant_decompress, grain_mask_decompress)

        return dec, bpp, partition_map

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, x, **kwargs):
        log = dict()

        x = x.to(self.device)
        xrec, _, grain_indices = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        # log["grain_map"] = draw_triple_grain_256res_color(images=x.clone(), indices=grain_indices, scaler=0.7)
        # x_entropy = x_entropy.sub(x_entropy.min()).div(max(x_entropy.max() - x_entropy.min(), 1e-5))
        # log["entropy_map"] = draw_dual_grain_256res_color(images=x.clone(), indices=x_entropy, scaler=0.7)
        log["partition_map"] = draw_triple_grain_256res(images=x.clone(), indices=grain_indices)
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    
    
class Entropy(nn.Sequential):
    def __init__(self, patch_size):
        super(Entropy, self).__init__()
        self.psize = patch_size
        # unfolding image to non overlapping patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.psize, self.psize), stride=self.psize)

    def entropy(self, values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, batch: int, h_num: int, w_num: int) -> torch.Tensor:
        """Function that calculates the entropy using marginal probability distribution function of the input tensor
            based on the number of histogram bins.
        Args:
            values: shape [BxNx1].
            bins: shape [NUM_BINS].
            sigma: shape [1], gaussian smoothing factor.
            batch: int, size of the batch
        Returns:
            torch.Tensor:
        """
        epsilon = 1e-40
        values = values.unsqueeze(2)
        residuals = values - bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
        pdf = pdf / normalization + epsilon
        entropy = - torch.sum(pdf * torch.log(pdf), dim=1)
        entropy = entropy.reshape((batch, -1))
        entropy = rearrange(entropy, "B (H W) -> B H W", H=h_num, W=w_num)
        return entropy

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = inputs.shape
        # number of patches per image and per dim
        patch_num = int(height * width // self.psize ** 2)
        h_num = int(height // self.psize)
        w_num = int(width // self.psize)

        gray_images = 0.2989 * inputs[:, 0:1, :, :] + 0.5870 * inputs[:, 1:2, :, :] + 0.1140 * inputs[:, 2:, :, :]

        # create patches of size (batch x patch_size*patch_size x h*w/ (patch_size*patch_size))
        unfolded_images = self.unfold(gray_images)
        # reshape to (batch * h*w/ (patch_size*patch_size) x (patch_size*patch_size)
        unfolded_images = unfolded_images.transpose(1, 2)
        unfolded_images = torch.reshape(unfolded_images.unsqueeze(2),
                                        (unfolded_images.shape[0] * patch_num, unfolded_images.shape[2]))

        entropy = self.entropy(unfolded_images, bins=torch.linspace(-1, 1, 32).to(device=inputs.device),
                               sigma=torch.tensor(0.01), batch=batch_size, h_num=h_num, w_num=w_num)

        return entropy
