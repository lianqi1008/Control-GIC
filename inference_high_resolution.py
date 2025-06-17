import argparse
import os
import pathlib
import pickle
import sys

import yaml
sys.path.append(".")
sys.path.append("CGIC/models/model")

import torch
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import json
from omegaconf import OmegaConf
from numpy import pi, exp, sqrt

import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.nn.functional import mse_loss, l1_loss
from torch.utils.data import DataLoader, Dataset

from CGIC.models.model import CGIC
from CGIC.tools.indices_coding import HuffmanCoding as HuffmanCoding # Huffman with frequency
from CGIC.tools.mask_coding import BinaryCoding

class ImageDataset(Dataset):
    def __init__(
        self,
        imagenet_images_dir: Union[str, Path],
        target_size: int = 512,
        images_range: Optional[Tuple[int, int]] = (0, -1)
    ) -> None:
        super().__init__()
        self.target_size = target_size

        imagenet_images_dir = Path(imagenet_images_dir)
        self.image_paths = sorted([p for p in imagenet_images_dir.glob('**/*') if self._is_image_path(p)])
        if images_range[1] > 0:
            self.image_paths = self.image_paths[images_range[0]:images_range[1]]
        print(f'Found {len(self.image_paths)} images to reconstruct')

        self.transform = T.Compose([
            # T.RandomResizedCrop(self.target_size, scale=(1., 1.), ratio=(1., 1.), interpolation=Image.Resampling.BICUBIC),
            T.ToTensor()
        ])
    
    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index])
        # image = image.convert('RGB')
        image = self._resize_and_crop(image)
        image = self.transform(image)
        return image
    
    def _resize_and_crop(self, img):
        w, h  = img.size
        hn = h // 16
        wn = w // 16
        img = TF.center_crop(img, output_size=[int(16*hn), int(wn*16)])
        return img

    def __len__(self) -> int:
        return len(self.image_paths)

    def _is_image_path(self, path: Path) -> bool:
        ext = path.name[path.name.rfind('.')+1:]
        if (path.is_dir()
                or path.name.startswith('.')
                or len(ext) == 0
                or ext.lower() not in ('jpg', 'jpeg', 'png')):
            return False
        return True

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_model(config, ckpt_path=None):
    model = CGIC(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Restored from {ckpt_path}")
    return model.eval()

def write_images(
    images: torch.Tensor,
    output_dir: Path,
    i: int,
    batch_size: int,
    start_offset: int,
    bpp: None,
) -> None:
    images = (255 * images.permute(0, 2, 3, 1).detach().cpu().numpy()).astype(np.uint8)
    for j, img in enumerate(images):
        k = i * batch_size + j + start_offset
        img = Image.fromarray(img)
        if bpp!=None:
            img.save(output_dir / f'{k:03d}_{bpp:05f}.png')
        else:
            img.save(output_dir / f'{k:03d}.png')

def nonoverlapping_grid_indices(x_padded):
    _, c, h, w = x_padded.shape
    hn = h//768
    wn = w//768
    h_list = [i for i in range(0, h, 768)]
    tile_h_list = [768 for i in range(hn)]
    if h%768!=0:
        tile_h_list.append(h%768)

    w_list = [i for i in range(0, w, 768)]
    tile_w_list = [768 for i in range(wn)]
    if w%768!=0:
        tile_w_list.append(w%768)
    return h_list, w_list, tile_h_list, tile_w_list

def _gaussian_weights(tile_width, tile_height, nbatches, device):
    """Generates a gaussian mask of weights for tile contributions"""

    latent_width = tile_width
    latent_height = tile_height

    var = 0.01
    # -1 because index goes from 0 to latent_width - 1
    midpoint = (latent_width - 1) / 2
    x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width) /
                   (2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height) /
                   (2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device=device), (nbatches, 3, 1, 1))

def compute_padding(in_h: int, in_w: int, *, out_h=None, out_w=None, min_div=1):
    """Returns tuples for padding and unpadding.

    Args:
        in_h: Input height.
        in_w: Input width.
        out_h: Output height.
        out_w: Output width.
        min_div: Length that output dimensions should be divisible by.
    """
    if out_h is None:
        out_h = (in_h + min_div - 1) // min_div * min_div
    if out_w is None:
        out_w = (in_w + min_div - 1) // min_div * min_div

    if out_h % min_div != 0 or out_w % min_div != 0:
        raise ValueError(
            f"Padded output height and width are not divisible by min_div={min_div}."
        )

    left = (out_w - in_w) // 2
    right = out_w - in_w - left
    top = (out_h - in_h) // 2
    bottom = out_h - in_h - top

    pad = (left, right, top, bottom)
    unpad = (-left, -right, -top, -bottom)

    return pad, unpad


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument('-i', '--images_dir', type=str, default='',
                        required=False, help='Path to the root directory where the images are')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=1, help='Number of images in a minibatch')
    parser.add_argument('-n', '--num_workers', type=int, default=1,
                        help='Number of worker threads to load the images')
    parser.add_argument('-s', '--image_size', type=int,
                        default=512, help='Size of the reconstructed image')
    parser.add_argument('-o', '--output_dir', type=str, default='output_reconstruction',
                        help='Path to a directory where the outputs will be saved')
    parser.add_argument('-w', '--write_partiton_map', default=False,
                        help='If set, the partition maps will also be saved to the output directory')
    parser.add_argument('-r', '--images_range', type=int, nargs=2, default=(0, -1),
                        help=('Optional. To use provide two values indicating the starting and ending indices of the images to be loaded.'
                              ' Only images within this range will be loaded. Useful for manually sharding the dataset if running all images'
                              ' at once would take too much time.'))
    return parser


def main():
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    dataset = ImageDataset(opt.images_dir, opt.image_size, opt.images_range)
    dataloader = DataLoader(dataset, opt.batch_size, num_workers=opt.num_workers)

    config = load_config("./configs/config_inference.yaml", display=True)
    model = load_model(config).to('cuda')
    
    frequency = model.quantize.embedding_counter

    h_string = HuffmanCoding(frequency)
    h_mask = BinaryCoding()
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    rec_output_dir = Path(opt.output_dir) / 'reconstructed'
    rec_output_dir.mkdir(parents=True, exist_ok=True)

    bpp_sum = 0.
    with open(os.path.join(opt.output_dir, 'bpp.txt'), 'a') as f:
        for k, x in enumerate(tqdm(dataloader)):
            bit_sum = 0.0

            if torch.cuda.is_available():
                x = x.cuda()

            h, w = x.shape[-2], x.shape[-1]
            pad, unpad = compute_padding(h, w, min_div=2**4)
            x_padded = F.pad(x, pad, mode="constant", value=0)
            h_list, w_list, tile_height_size_list, tile_width_size_list = nonoverlapping_grid_indices(x_padded)

            x_rec = torch.zeros(x_padded.shape, device=x.device)
            contributors = torch.zeros(x_padded.shape, device=x.device)

            count = 0
            for i in range(len(h_list)):
                for j in range(len(w_list)):
                    count += 1
                    hi = h_list[i]
                    wi = w_list[j]
                    tile_hight_size = tile_height_size_list[i]
                    tile_width_size = tile_width_size_list[j]
                    tile_weights = _gaussian_weights(tile_width_size, tile_hight_size, 1, x_padded.device)
                    # print(f'{count}/{len(h_list)*len(w_list)}')
                    
                    with torch.no_grad():
                        x_tile_rec, bpp, _ = model.compress(x_padded[:, :, hi:hi + tile_hight_size, wi:wi + tile_width_size], opt.output_dir, h_string, h_mask, False)
                        
                    x_rec[:, :, hi:hi + tile_hight_size, wi:wi + tile_width_size] += x_tile_rec * tile_weights
                    contributors[:, :, hi:hi + tile_hight_size, wi:wi + tile_width_size] += tile_weights
                    bit_sum += bpp*tile_width_size*tile_hight_size
                    # print(f'tile: {tile_width_size}*{tile_hight_size} \t bpp: {bpp}')
                    
            x_rec /= contributors
            x_rec = x_rec.clamp(0, 1)
            x_rec = F.pad(x_rec, unpad)
            bpp_image = bit_sum/x.shape[-1]/x.shape[-2]
            write_images(x_rec, rec_output_dir, k, opt.batch_size, opt.images_range[0], bpp=bpp_image)
            
            bpp_sum += bpp_image
            f.write(f'image: {k} \t bpp: {bpp_image}\n')
            # print(f'Bpp: {bpp_image}')
        f.write(f'Bpp Average: {bpp_sum/len(dataset)}')
        print(f'Bpp Average: {bpp_sum/len(dataset)}')
    f.close()


if __name__ == '__main__':
    main()
