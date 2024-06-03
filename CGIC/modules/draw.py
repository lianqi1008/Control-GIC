import torch
import importlib
import torchvision
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from einops import rearrange
import numpy as np

transform_PIL = transforms.Compose([transforms.ToPILImage()])

color_dict = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "blue": (5, 39, 175),
}

# same function in torchvision.utils.save_image(normalize=True)
def image_normalize(tensor, value_range=None, scale_each=False):
    tensor = tensor.clone()
    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    def norm_range(t, value_range):
        if value_range is not None:
            norm_ip(t, value_range[0], value_range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    if scale_each is True:
        for t in tensor:  # loop over mini-batch dimension
            norm_range(t, value_range)
    else:
        norm_range(tensor, value_range)
    
    return tensor


def draw_triple_grain_256res_color(images=None, image_size=256, indices=None, low_color="blue", high_color="red", scaler=0.9):
    # indices: [batch_size, height, weight]
    # 0 for coarse-grained, 1 for median-grained, 2 for fine grain
    height = images.size(-2)
    weight = images.size(-1)
    if images is None:
        images = torch.ones(indices.size(0), 3, weight, height)
    indices = indices.unsqueeze(1)
    size_w = weight // indices.size(-1)
    size_h = height // indices.size(-1)

    indices = indices.repeat_interleave(size_w, dim=-1).repeat_interleave(size_h, dim=-2)
    indices = indices / 2
    
    bs = images.size(0)

    low = Image.new('RGB', (images.size(-2), images.size(-1)), color_dict[low_color])
    high = Image.new('RGB', (images.size(-2), images.size(-1)), color_dict[high_color])

    for i in range(bs):
        image_i_pil = transform_PIL(image_normalize(images[i]))

        score_map_i_np = rearrange(indices[i], "C H W -> H W C").cpu().detach().numpy()
        score_map_i_blend = Image.fromarray(
            np.uint8(high * score_map_i_np + low * (1 - score_map_i_np)))
        
        image_i_blend = Image.blend(image_i_pil, score_map_i_blend, scaler)

        if i == 0:
            blended_images = torchvision.transforms.functional.to_tensor(image_i_blend).unsqueeze(0)
        else:
            blended_images = torch.cat([
                blended_images, torchvision.transforms.functional.to_tensor(image_i_blend).unsqueeze(0)
            ], dim=0)
    return blended_images

def draw_triple_grain_256res(images=None, indices=None):
    # indices: [batch_size, height, weight]
    # 0 for coarse-grained, 1 for median-grained, 2 for fine grain
    height = images.size(-2)
    weight = images.size(-1)
    if images is None:
        images = torch.ones(indices.size(0), 3, height, weight)
    size_w = weight // indices.size(2)
    size_h = height // indices.size(1)
    for b in range(indices.size(0)): # batch_size
        for i in range(indices.size(1)//4):
            for j in range(indices.size(2)//4):  # draw coarse-grain line
                y_min = size_h * 4 * i # 0, 32, 64, 128, 
                y_max = size_h * 4 * (i + 1) # 0, 32, 64, 128, 
                x_min = size_w * 4 * j
                x_max = size_w * 4 * (j + 1)
                images[b, :, y_min, x_min:x_max] = -1
                images[b, :, y_min:y_max, x_min] = -1

    for b in range(indices.size(0)): # batch_size
        for i in range(indices.size(1)//2):
            for j in range(indices.size(2)//2):  # draw coarse-grain line
                if indices[b, i*2, j*2] == 1:  # draw median-grain line
                    y_min = size_h * 2 * i # 0, 32, 64, 128, 
                    y_max = size_h * 2 * (i + 1) # 0, 32, 64, 128, 
                    x_min = size_w * 2 * j
                    x_max = size_w * 2 * (j + 1)
                    images[b, :, y_min, x_min:x_max] = -1
                    images[b, :, y_min:y_max, x_min] = -1

    for b in range(indices.size(0)): # batch_size
        for i in range(indices.size(1)):
            for j in range(indices.size(2)):  # draw coarse-grain line
                if indices[b, i, j] == 2:  # draw fine-grain line
                    y_min = size_h * i # 0, 32, 64, 128, 
                    y_max = size_h * (i + 1) # 0, 32, 64, 128, 
                    x_min = size_w * j
                    x_max = size_w * (j + 1)
                    images[b, :, y_min, x_min:x_max] = -1
                    images[b, :, y_min:y_max, x_min] = -1

    return images

def instantiate_from_config(config):
    def get_obj_from_str(string, reload=False):
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))