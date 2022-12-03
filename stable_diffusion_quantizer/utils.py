import typing
import os
import torch
import torch.nn as nn
import cv2
import numpy as np


def remap_layers(model: nn.Module, layers_map: typing.Dict, params: typing.Dict) -> None:
    def _remap_layers(m: nn.Module):
        for key, val in m._modules.items():
            if len(val._modules):
                _remap_layers(val)
            elif type(val) in layers_map:
                m._modules[key] = layers_map[type(val)](m = val, **params)
    _remap_layers(model)


def get_layers_map(m: nn.Module, prefix: str="") -> typing.Tuple[str, type]:
    for key, val in m._modules.items():
        name = f"{prefix}.{key}" if prefix != "" else f"{key}"
        if len(val._modules):
            yield from get_layers_map(val, prefix=name)
        else:
            yield name, type(val)


def get_layer_by_name(m: nn.Module, name: str) -> nn.Module:
    for key in name.split('.'):
        m = m._modules[key]
    return m


def set_layer_by_name(m: nn.Module, name: str, layer: nn.Module) -> None:
    name = name.split('.')
    for key in name[:-1]:
        m = m._modules[key]
    m._modules[name[-1]] = layer


def get_names_by_type(m: nn.Module, dtype: type) -> nn.Module:
    out = []
    for name, layer_type in get_layers_map(m):
        if layer_type == dtype:
            out.append(name)
    return out


def image_to_tensor(img, bgr2rgb=True, normalize=True, vrange=(0.0, 255.0), image_size=None, return_tensors="pt"):
    if image_size is not None:
        h, w = img.shape[:2]
        if h != image_size[0] or w != image_size[1]:
            img = cv2.resize(img, image_size)
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # normalize
    if normalize:
        img = (img - vrange[0]) / (vrange[1] - vrange[0])
        img = 2.0 * img - 1.0
    img = img.transpose(2, 0, 1).astype(np.float32)
    if return_tensors == "pt":
        img = torch.from_numpy(img)
    return img
