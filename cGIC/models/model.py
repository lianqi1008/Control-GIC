import json
import os
import pathlib
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import partial

from cGIC.util import instantiate_from_config
from cGIC.models.utils import Scheduler_LinearWarmup
from cGIC.modules.vqvae.vqvae_blocks_0429_3grain import MOEncoder
from cGIC.modules.vqvae.quantize_triple_entropy_resblock import VectorQuantize2 as VectorQuantizer
from cGIC.modules.vqvae.movq_modules_0315_0429_3grain import MOVQDecoder
from cGIC.models.vqgan import MOVQ as MOVQModel
from cGIC.models.ema import LitEma
from cGIC.modules.util import disabled_train
from cGIC.modules.dynamic_modules.utils import draw_triple_grain_256res_color, draw_triple_grain_256res
# from cGIC.tools.indices_coding_str import zip_coding, huffman_coding
from timm.models.vision_transformer import Block

from einops import rearrange
import numpy as np

class cGIC(pl.LightningModule):
    def __init__(self,
                 encoderconfig,
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
                 entropy_patch_size=16
                 ):
        super().__init__()

        self.image_key = image_key
        self.encoder = MOEncoder(**encoderconfig)
        self.decoder = MOVQDecoder(zq_ch=embed_dim, **ddconfig)
        if lossconfig is not None:
            self.loss = instantiate_from_config(lossconfig)
        if learning_rate is not None:
            self.learning_rate = learning_rate
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.entropy_calculation_p8 = Entropy(8)
        self.entropy_calculation_p8 = self.entropy_calculation_p8.eval()
        self.entropy_calculation_p8.train = disabled_train

        self.entropy_calculation_p16 = Entropy(16)
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

        sd = ckpt["state_dict"]
        # sd = ckpt

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

        h = self.quant_conv(h)
        quant, emb_loss, ind = self.quantize(h)

        return quant, emb_loss, grain_indices, grain_mask, ind, fine_ratio

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
        quant, diff, grain_indices, grain_mask, _, _ = self.encode(input)
        dec = self.decode(quant, grain_mask)
        return dec, diff, grain_indices

    def compress(self, input, path, h_string, h_mask):
        assert len(input.shape)==4
        # compress
        quant, diff, grain_indices, grain_mask, ind, _ = self.encode(input)
        partiton_map = draw_triple_grain_256res(images=input.clone(), indices=grain_indices)
        ind = ind.view(-1, quant.shape[-2], quant.shape[-1])

        # ind_coarse = (ind*grain_mask[:,0,:,:]).view(-1, ind.shape[-2]//4, 4, ind.shape[-1]//4, 4)
        # ind_coarse[:,:,1:4,:,1:4] = -1

        # ind_medium = (ind*grain_mask[:,1,:,:]).view(-1, ind.shape[-2]//2, 2, ind.shape[-1]//2, 2)
        # ind_medium[:,:,1:2,:,1:2] = -1

        # ind_fine = ind*grain_mask[:,2,:,:]

        # ind_final = ind_coarse + ind_medium + ind_fine 

        # ind_coarse_4X = ind*grain_mask.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)[:,0,:,:]
        ind_coarse = ind[:,::4,::4][grain_mask[0] == 1]

        # ind_medium_2X = ind*grain_mask.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)[:,1,:,:]
        # ind_medium = ind_medium_2X[:,::2,::2].flatten().tolist()

        ind_medium = ind[:,::2,::2][grain_mask[1] == 1]

        # ind_fine = ind*grain_mask.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)[:,1,:,:]
        # 获取非零元素的索引
        ind_fine = ind[grain_mask[2] == 1]

        # frequency = self.quantize.embedding_counter
        # frequency_sorted = sorted(frequency.items(),  key=lambda d: d[1], reverse=True)
        # new_value = frequency_sorted[0][1] + nn.Parameter(torch.ones(1))
        # frequency.add_module('-1', new_value)

        # # Huffman with frequency
        # from cGIC.tools.indices_coding_with_frequency_pre_optimz import HuffmanCoding as HuffmanCoding
        # from cGIC.tools.indices_coding_original import HuffmanCoding as HuffmanCoding_original

        num_pixels = input.shape[2]*input.shape[3]
        # h_string = HuffmanCoding(frequency)
        string_coarse = h_string.compress(ind_coarse, os.path.join(path, 'string_coarse.bin')) 
        string_medium = h_string.compress(ind_medium, os.path.join(path, 'string_medium.bin'))
        string_fine = h_string.compress(ind_fine, os.path.join(path, 'string_fine.bin'))

        # h_mask = HuffmanCoding_original()
        string_mask_coarse = h_mask.compress(grain_mask[0].flatten(), os.path.join(path, 'mask_coarse.bin'))
        string_mask_medium = h_mask.compress(grain_mask[1].flatten(), os.path.join(path, 'mask_medium.bin'))
        # mask_fine = h.compress(grain_mask[2].flatten(), os.path.join(path, 'mask_fine.bin'))

        bpp = (os.path.getsize(string_coarse) + os.path.getsize(string_medium) + os.path.getsize(string_fine) + os.path.getsize(string_mask_coarse) + os.path.getsize(string_mask_medium)) * 8 / num_pixels
        # bpp = (os.path.getsize(string_coarse) + os.path.getsize(string_medium) + os.path.getsize(string_mask_coarse)) * 8 / num_pixels
        # bpp = (os.path.getsize(string_fine) + os.path.getsize(string_medium) + os.path.getsize(string_mask_medium)) * 8 / num_pixels

        
        # print('\n')
        # print(f'string_coarse: {os.path.getsize(string_coarse)} byte')
        # print(f'string_medium: {os.path.getsize(string_medium)} byte')
        # print(f'string_fine: {os.path.getsize(string_fine)} byte')
        # print(f'string_mask_coarse: {os.path.getsize(string_mask_coarse)} byte')
        # print(f'string_mask_medium: {os.path.getsize(string_mask_medium)} byte')
        
        # strings = {
        #     "string_coarse": string_coarse,
        #     "string_medium": string_medium,
        #     "string_fine": string_fine,
        #     "mask_coarse": mask_coarse,
        #     "mask_medium": mask_medium,
        #     "mask_fine": mask_fine,
        # }
        # bpp = sum(len(s[0]) for string in strings for s in string) * 8.0 / num_pixels
        # bpp = sum(len(string) for string in strings.values()) * 8.0 / num_pixels

        # decompress
        # ind_decompress = torch.zeros_like(ind)
        ind_coarse_decompress = h_string.decompress_string(os.path.join(path, 'string_coarse.bin'))
        ind_medium_decompress = h_string.decompress_string(os.path.join(path, 'string_medium.bin'))
        ind_fine_decompress = h_string.decompress_string(os.path.join(path, 'string_fine.bin'))

        if ind_coarse_decompress:
            mask_coarse_decompress = h_mask.decompress_string(os.path.join(path, 'mask_coarse.bin'))
            # ind_decompress = torch.zeros_like(torch.tensor(mask_coarse_decompress))
            # ind_decompress = ind_decompress.view(1, int(len(ind_decompress)**0.5), int(len(ind_decompress)**0.5))
            # ind_decompress = ind_decompress.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)

            ind_coarse_reshape = []
            j = 0
            for i, value in enumerate(mask_coarse_decompress):
                if value == 1:
                    ind_coarse_reshape.append(ind_coarse_decompress[j])
                    j += 1
                else:
                    ind_coarse_reshape.append(0)
            ind_coarse_reshape = torch.tensor(ind_coarse_reshape).view(1, ind.shape[-2]//4, ind.shape[-1]//4)
            ind_coarse_reshape = ind_coarse_reshape.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2).int().to(ind.device)
            mask_coarse_reshape = torch.tensor(mask_coarse_decompress).view(1, ind.shape[-2]//4, ind.shape[-1]//4)
            mask_coarse_reshape = mask_coarse_reshape.int().to(ind.device)

        else:
            mask_coarse_reshape = torch.zeros(1, ind.shape[-2]//4, ind.shape[-1]//4).int().to(ind.device)
            ind_coarse_reshape = torch.zeros(ind.shape).int().to(ind.device)
            
        if ind_medium_decompress:
            mask_medium_decompress = h_mask.decompress_string(os.path.join(path, 'mask_medium.bin'))
            ind_medium_reshape = []

            j = 0
            for i, value in enumerate(mask_medium_decompress):
                if value == 1:
                    ind_medium_reshape.append(ind_medium_decompress[j])
                    j += 1
                else:
                    ind_medium_reshape.append(0)
            ind_medium_reshape = torch.tensor(ind_medium_reshape).view(1, ind.shape[-2]//2, ind.shape[-1]//2)
            ind_medium_reshape = ind_medium_reshape.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2).int().to(ind.device)
            mask_medium_reshape = torch.tensor(mask_medium_decompress).view(1, ind.shape[-2]//2, ind.shape[-1]//2)
            mask_medium_reshape = mask_medium_reshape.int().to(ind.device)
        else:
            mask_medium_reshape = torch.zeros(1, ind.shape[-2]//2, ind.shape[-1]//2).int().to(ind.device)
            ind_medium_reshape = torch.zeros(ind.shape).int().to(ind.device)
        
        if ind_fine_decompress:
            mask_fine_reshape = (1 - mask_medium_reshape.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2) - mask_coarse_reshape.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2))
            ind_fine_reshape = []

            k = 0
            for i in range(mask_fine_reshape.shape[-2]):
                for j in range(mask_fine_reshape.shape[-1]):
                    if mask_fine_reshape[:,i,j] == 1:
                        ind_fine_reshape.append(ind_fine_decompress[k])
                        k += 1
                    else:
                        ind_fine_reshape.append(0)
            ind_fine_reshape = torch.tensor(ind_fine_reshape).view(1, ind.shape[-2], ind.shape[-1]).int().to(ind.device)
            ind_decompress = ind_fine_reshape + ind_medium_reshape + ind_coarse_reshape
        else:
            ind_decompress = ind_medium_reshape + ind_coarse_reshape
            
        mask_fine_reshape = (1 - mask_medium_reshape.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2) - mask_coarse_reshape.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2))
        grain_mask_decompress = [mask_coarse_reshape, mask_medium_reshape, mask_fine_reshape]

        if torch.max(ind_decompress)>=16384 or torch.min(ind_decompress)<0:
            print(f'Input MAX: {torch.max(ind_decompress)} Input MIN: {torch.min(ind_decompress)}')
        quant_decompress = self.quantize.embedding(ind_decompress.flatten()).view(ind_decompress.shape[0], ind_decompress.shape[1], ind_decompress.shape[2], -1)
        quant_decompress = rearrange(quant_decompress, 'b h w c -> b c h w')

        dec = self.decode(quant_decompress, grain_mask_decompress)

        return dec, bpp, partiton_map

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.size(1) != 3:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            # self.ema_encoder(self.encoder)
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
        # sd = torch.load('./checkpoint/67M/cGIC_67M.ckpt', map_location="cpu")
        # keys = list(sd.keys())
        
        # selected_params = []
        # print("Params Optimized in Autoencoder:")
        # for name, param in self.named_parameters():
        #     if name not in keys and 'perceptual' not in name:
        #         print(name)
        #         param.requires_grad = True
        #         selected_params.append(param)

        # opt_ae = torch.optim.Adam(selected_params,
        #                           lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        # warmup_steps = 75006
        # scheduler_ae = {
        #     "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_ae, Scheduler_LinearWarmup(warmup_steps)), "interval": "step", "frequency": 1,
        # }
        # scheduler_disc = {
        #     "scheduler": torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps)), "interval": "step", "frequency": 1,
        # }
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, x, **kwargs):
        log = dict()
        # x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _, grain_indices = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        log["grain_map"] = draw_triple_grain_256res_color(images=x.clone(), indices=grain_indices, scaler=0.7)
        # x_entropy = x_entropy.sub(x_entropy.min()).div(max(x_entropy.max() - x_entropy.min(), 1e-5))
        # log["entropy_map"] = draw_dual_grain_256res_color(images=x.clone(), indices=x_entropy, scaler=0.7)
        log["partition_map"] = draw_triple_grain_256res(images=x.clone(), indices=grain_indices)
        # log["feature_map"] = z_feature
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
    
# class Entropy(nn.Sequential):
#     def __init__(self, patch_size, image_width, image_height):
#         super(Entropy, self).__init__()
#         self.width = image_width
#         self.height = image_height
#         self.psize = patch_size
#         # number of patches per image
#         self.patch_num = int(self.width * self.height / self.psize ** 2)
#         self.hw = int(self.width // self.psize)
#         # unfolding image to non overlapping patches
#         self.unfold = torch.nn.Unfold(kernel_size=(self.psize, self.psize), stride=self.psize)

#     def entropy(self, values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor, batch: int) -> torch.Tensor:
#         """Function that calculates the entropy using marginal probability distribution function of the input tensor
#             based on the number of histogram bins.
#         Args:
#             values: shape [BxNx1].
#             bins: shape [NUM_BINS].
#             sigma: shape [1], gaussian smoothing factor.
#             batch: int, size of the batch
#         Returns:
#             torch.Tensor:
#         """
#         epsilon = 1e-40
#         values = values.unsqueeze(2)
#         residuals = values - bins.unsqueeze(0).unsqueeze(0)
#         kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

#         pdf = torch.mean(kernel_values, dim=1)
#         normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
#         pdf = pdf / normalization + epsilon
#         entropy = - torch.sum(pdf * torch.log(pdf), dim=1)
#         entropy = entropy.reshape((batch, -1))
#         entropy = rearrange(entropy, "B (H W) -> B H W", H=self.hw, W=self.hw)
#         return entropy

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         batch_size = inputs.shape[0]
#         gray_images = 0.2989 * inputs[:, 0:1, :, :] + 0.5870 * inputs[:, 1:2, :, :] + 0.1140 * inputs[:, 2:, :, :]

#         # create patches of size (batch x patch_size*patch_size x h*w/ (patch_size*patch_size))
#         unfolded_images = self.unfold(gray_images)
#         # reshape to (batch * h*w/ (patch_size*patch_size) x (patch_size*patch_size)
#         unfolded_images = unfolded_images.transpose(1, 2)
#         unfolded_images = torch.reshape(unfolded_images.unsqueeze(2),
#                                         (unfolded_images.shape[0] * self.patch_num, unfolded_images.shape[2]))

#         entropy = self.entropy(unfolded_images, bins=torch.linspace(-1, 1, 32).to(device=inputs.device),
#                                sigma=torch.tensor(0.01), batch=batch_size)

#         return entropy