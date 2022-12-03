import torch.nn as nn
from stable_diffusion_quantizer.modules import Calibrator, QConv2d, QLinear
from stable_diffusion_quantizer.utils import *


def get_names(model, conv=True, linear=True, exclude=[], **kwargs):
    names = []
    if conv:
        names.extend(get_names_by_type(model, nn.Conv2d))
    if linear:
        names.extend(get_names_by_type(model, nn.Linear))
    names = list(set(names) - set(exclude))
    return names


def modules_to_calibrator(model, names, momentum=0.1, **kwargs):
    for name in names:
        src = get_layer_by_name(model, name)
        dst = Calibrator(src, momentum=momentum)
        set_layer_by_name(model, name, dst)


def modules_to_qmodules(model, names, bits=8, **kwargs):
    for name in names:
        src = get_layer_by_name(model, name)
        if type(src) == nn.Conv2d:
            dst = QConv2d(src, bits=bits)
        elif type(src) == nn.Linear:
            dst = QLinear(src, bits=bits)
        else:
            raise ValueError("Unknown module type")
        set_layer_by_name(model, name, dst)


def calibrator_to_qmodule(model, names, bits=8, **kwargs):
    for name in names:
        src = get_layer_by_name(model, name)
        if src.mtype() == nn.Conv2d:
            dst = QConv2d(src.m, bits=bits)
        elif src.mtype() == nn.Linear:
            dst = QLinear(src.m, bits=bits)
        else:
            raise ValueError("Unknown module type")
        dst.activation_quantizer.lower.data.fill_(src.params["min"])
        dst.activation_quantizer.length.data.fill_(src.params["max"] - src.params["min"])
        dst.weight_quantizer.scale.data.fill_(src.m.weight.abs().max())
        set_layer_by_name(model, name, dst)
