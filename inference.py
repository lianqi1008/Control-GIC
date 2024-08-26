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

import torchvision
import torchvision.transforms as T
from torch.nn.functional import mse_loss, l1_loss
from torch.utils.data import DataLoader, Dataset

from CGIC.models.model import CGIC
import torchvision.transforms.functional as TF

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
        # print(f'Image Resize to {self.target_size} × {self.target_size}')

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
        hn = h // 32
        wn = w // 32
        img = TF.center_crop(img, output_size=[int(32*hn), int(wn*32)])
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
        
def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument('-i', '--images_dir', type=str, default='../dataset/Kodak', required=False, help='Path to the root directory where the images are')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Number of images in a minibatch')
    parser.add_argument('-n', '--num_workers', type=int, default=1, help='Number of worker threads to load the images')
    parser.add_argument('-s', '--image_size', type=int, default=512, help='Size of the reconstructed image')
    parser.add_argument('-o', '--output_dir', type=str, default='output', help='Path to a directory where the outputs will be saved')
    parser.add_argument('-w', '--write_output_image', action='store_true', help='If set, the reconstructed images will be saved to the output directory')
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

    config = load_config("./configs/config_inference.yaml", display=False)
    print(config)
    model = load_model(config).to('cuda')
    
    frequency = model.quantize.embedding_counter

    from CGIC.tools.indices_coding import HuffmanCoding as HuffmanCoding # Huffman with frequency
    from CGIC.tools.mask_coding import BinaryCoding

    h_string = HuffmanCoding(frequency)
    h_mask = BinaryCoding()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    if opt.write_output_image:
        rec_output_dir = Path(opt.output_dir) / 'output'
        rec_output_dir.mkdir(parents=True, exist_ok=True)
        partition_output_dir = Path(opt.output_dir) / 'partition_map'
        partition_output_dir.mkdir(parents=True, exist_ok=True)

    bpp_sum = 0.
    
    with open(os.path.join(opt.output_dir, 'bpp.txt'),'a') as f:
        for i, x in enumerate(tqdm(dataloader)):
            if torch.cuda.is_available():
                x = x.cuda()
            
            with torch.no_grad():
                if opt.write_output_image:
                    x_rec, bpp, partition_map = model.compress(x, opt.output_dir, h_string, h_mask, save_img=True)
                    x_rec = x_rec.clamp(0, 1)
                    write_images(x_rec, rec_output_dir, i, opt.batch_size, opt.images_range[0], bpp=bpp)
                    write_images(partition_map, partition_output_dir, i, opt.batch_size, opt.images_range[0], None)
                else:
                    bpp = model.compress(x, opt.output_dir, h_string, h_mask, save_img=False)

            bpp_sum += bpp
            f.write(f'image: {i} \t bpp: {bpp}\n')
        f.write(f'Bpp Average: {bpp_sum/len(dataset)}')
        print(f'Bpp Average: {bpp_sum/len(dataset)}')
    f.close()


if __name__ == '__main__':
    main()
