import paddle

import paddle
import cv2
import numpy as np
import paddle.vision
import paddle.optimizer
import os

import model
import numpy as np

from PIL import Image
import glob
import time

import pathlib
import paddle
import warnings
import math
import numpy as np
from PIL import Image
from typing import Union, Optional, List, Tuple, Text, BinaryIO


@paddle.no_grad()
def make_grid(tensor: Union[paddle.Tensor, List[paddle.Tensor]],
              nrow: int=8,
              padding: int=2,
              normalize: bool=False,
              value_range: Optional[Tuple[int, int]]=None,
              scale_each: bool=False,
              pad_value: int=0,
              **kwargs) -> paddle.Tensor:
    if not (isinstance(tensor, paddle.Tensor) or
            (isinstance(tensor, list) and all(
                isinstance(t, paddle.Tensor) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clip(min=low, max=high)
            img = img - low
            img = img / max(high - low, 1e-5)

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

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] +
                                                        padding)
    num_channels = tensor.shape[1]
    grid = paddle.full((num_channels, height * ymaps + padding,
                        width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y + 1) * height, x * width + padding:(
                x + 1) * width] = tensor[k]
            k = k + 1
    return grid


@paddle.no_grad()
def save_image(tensor: Union[paddle.Tensor, List[paddle.Tensor]],
               fp: Union[Text, pathlib.Path, BinaryIO],
               format: Optional[str]=None,
               **kwargs) -> None:
    grid = make_grid(tensor, **kwargs)
    ndarr = paddle.clip(grid * 255 + 0.5, 0, 255).transpose(
        [1, 2, 0]).cast("uint8").numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)

def lowlight(image_path):
    device = paddle.get_device()
    data_lowlight = Image.open(image_path)
    data_lowlight = np.asarray(data_lowlight) / 255.0
    data_lowlight = paddle.to_tensor(data_lowlight, dtype='float32')
    data_lowlight = data_lowlight.transpose([2, 0, 1])
    data_lowlight = data_lowlight.unsqueeze(0)
    DCE_net = model.enhance_net_nopool()
    DCE_net.set_state_dict(paddle.load('snapshots/Epoch0.pdiparams'))
    DCE_net.to(device)
    start = time.time()
    e, enhanced_image, _ = DCE_net(data_lowlight)
    end_time = time.time() - start
    print(end_time)
    image_path = image_path.replace('test_data', 'result')
    if not os.path.exists(image_path.replace('/' + image_path.split('/')[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split('/')[-1], ''))

    save_image(enhanced_image, image_path)

if __name__ == '__main__':
    if paddle.is_compiled_with_cuda():
        paddle.set_device('gpu:0')
        print("Using CUDA device.")
    else:
        paddle.set_device('cpu')
        print("CUDA device not found. Using CPU.")

    with paddle.no_grad():
        filePath = 'data/test_data/'
        result_path = 'data/result/'
        file_list = os.listdir(filePath)
        for file_name in file_list:
            test_list = glob.glob(filePath + file_name + '/*')
            result_filename_path = filePath.replace('test_data', 'result') + file_name
            if not os.path.exists(result_filename_path):
                os.makedirs(result_filename_path)
            for i in range(len(test_list)):
                image = test_list[i]
                lowlight(image)


