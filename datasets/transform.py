import math
import numpy as np
import torch

def random_short_side_scale_jitter(images, min_size, max_size):
    size = int(round(np.random.uniform(min_size,max_size)))
    height = images.shape[2]
    width = images.shape[3]
    if(width <= height and width == size) or (
        height <= width and height == size
    ):
        return images

    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height)/width)*size))
    else:
        new_width = int(math.floor((float(width)/ height)*size))
    
    return torch.nn.functional.interpolate(
        images,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )

def random_crop(images, size):
    if images.shape[2] == size and images.shape[3] == size:
        return images

    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]
    return cropped

def horizontal_flip(prob, images):
    if np.random.uniform() < prob:
        images = images.flip((-1))
    return images

def uniform_crop(images, size, spatial_idx):
    assert spatial_idx in [0, 1, 2]
    height = images.shape[2]
    width = images.shape[3]

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]
    return cropped