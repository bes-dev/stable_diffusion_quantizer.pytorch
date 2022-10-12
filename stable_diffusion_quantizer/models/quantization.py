"""
Copyright 2022 by Sergei Belousov
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import math
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from .functional import straight_through_estimator, logit, binary_threshold
from torch.nn.common_types import _size_2_t

class StraightThroughEstimator(nn.Module):
    """ Implementation of Straight Through Estimator.
        Arguments:
            op (typing.Callable): the operation that applies to input tensor.
                                  The operation should have a similar shape of the input and output.
    """
    def __init__(self, op: typing.Callable) -> None:
        super().__init__()
        self.op = op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward path.
        Arguments:
            input (torch.Tensor): input tensor.

        Returns:
            output (torch.Tensor): output tensor.
        """
        return straight_through_estimator.apply(x, self.op)


class SymmetricQuantizer(nn.Module):
    """ Implementation of zero-centered symmetric quantization.
        Arguments:
            shape (typing.List[int]): shape of the quantization parameters.
            bits (int): precision of the quantization.
    """
    def __init__(self, shape: typing.List[int] = (1,), bits: int = 8) -> None:
        super().__init__()
        # parameters
        self.scale = nn.Parameter(torch.ones(shape), requires_grad = True)
        # utils
        self.bits = bits
        self.val_min = - 2 ** (bits - 1)
        self.val_max = 2 ** (bits - 1) - 1
        self.round_ste = StraightThroughEstimator(op=torch.round)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Fake quantize of the input tensor.
        Arguments:
            input (torch.Tensor): input tensor.

        Returns:
            output (torch.Tensor): quantized output tensor.
        """
        scale = torch.div(self.val_max, self.scale)
        clamp = torch.clamp(input * scale, min=self.val_min, max=self.val_max)
        quant = self.round_ste(clamp)
        quant_fake = quant * scale.reciprocal()
        return quant_fake

    def __repr__(self):
        return f"SymmetricQuantization(scale={self.scale.size()})"


class NonsymmetricQuantizerU(nn.Module):
    """ Implementation of nonsymmetric quantization to unsigned int.
        Arguments:
            shape (typing.List[int]): shape of the quantization parameters.
            bits (int): precision of the quantization.
    """
    def __init__(self, shape: typing.List[int] = (1,), bits: int = 8) -> None:
        super().__init__()
        # parameters
        self.lower = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.length = nn.Parameter(torch.ones(shape), requires_grad=True)
        # utils
        self.bits = bits
        self.round_ste = StraightThroughEstimator(op=torch.round)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Fake quantize of the input tensor.
        Arguments:
            input (torch.Tensor): input tensor.

        Returns:
            output (torch.Tensor): quantized output tensor.
        """
        length = self.length.abs()
        upper = self.lower + length
        scale = torch.div(length, 2 ** self.bits - 1)
        clamp = torch.div(torch.clamp(input, min=self.lower, max=upper) - self.lower, scale)
        quant = self.round_ste(clamp)
        quant_fake = scale * quant + self.lower
        return quant_fake

    def __repr__(self):
        return f"NonsymmetricQuantizerU(lower={self.lower.size()}, length=(self.length.size()))"


class NonsymmetricQuantizerS(nn.Module):
    """ Implementation of nonsymmetric quantization to signed int.
        Arguments:
            shape (typing.List[int]): shape of the quantization parameters.
            bits (int): precision of the quantization.
    """
    def __init__(self, shape: typing.List[int] = (1,), bits: int = 8) -> None:
        super().__init__()
        # parameters
        self.lower = nn.Parameter(torch.zeros(shape), requires_grad=True)
        self.length = nn.Parameter(torch.ones(shape), requires_grad=True)
        # utils
        self.bits = bits
        self.round_ste = StraightThroughEstimator(op=torch.round)
        self.val_min = - 2 ** (bits - 1)
        self.val_max = 2 ** (bits - 1) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Fake quantize of the input tensor.
        Arguments:
            input (torch.Tensor): input tensor.

        Returns:
            output (torch.Tensor): quantized output tensor.
        """
        length = self.length.abs()
        scale = (2 ** self.bits - 1) / length
        zero_point = - self.ste(self.lower * scale) - 2 ** (self.bits - 1)
        quant = torch.clamp(self.round_ste(scale * x + zero_point), min=self.val_min, max=self.val_max)
        quant_fake = (quant - zero_point) / scale
        return quant_fake

    def __repr__(self):
        return f"NonsymmetricQuantizationS(lower={self.lower.size()}, length={self.length.size()})"


class QModule(nn.Module):
    """ Base class for quantized modules.
    Arguments:
        m (nn.Module): source module to quantize.
        bits (int): precision of the quantization.
        activation_shape (List[int]): shape of quantization parameters for activation.
        weight_shape (List[int]): shape of quantization parameters for weight.
    """
    def __init__(
            self,
            m: nn.Module,
            bits: int,
            activation_shape: typing.List[int] = (1, ),
            weight_shape: typing.List[int] = (1, )
    ) -> None:
        super().__init__()
        self.m = m
        self.activation_quantizer = NonsymmetricQuantizerU(shape = activation_shape, bits = bits)
        self.weight_quantizer = SymmetricQuantizer(shape = weight_shape, bits = bits)

    def forward(self, input) -> torch.Tensor:
        """ Forward path.
        Arguments:
            input (torch.Tensor): input tensor.

        Returns:
            output (torch.Tensor): result of quantized operation.
        """
        input = self.activation_quantizer(input)
        weight = self.weight_quantizer(self.m.weight)
        return self._forward(input, weight, self.m.bias)

    def _forward(
            self,
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor
    ) -> torch.Tensor:
        """ Custom inference for quantized module.
        Arguments:
            input (torch.Tensor): input tensor.
            weight (torch.Tensor): input weight.
            bias (torch.Tensor): input bias.

        Returns:
            output (torch.Tensor): result of quantized operation.
        """
        raise NotImplemented


class QConv2d(QModule):
    def _forward(
            self,
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor
    ) -> torch.Tensor:
        """ Custom inference for quantized nn.Conv2d.
        Arguments:
            input (torch.Tensor): input tensor.
            weight (torch.Tensor): input weight.
            bias (torch.Tensor): input bias.

        Returns:
            output (torch.Tensor): result of quantized operation.
        """
        return self.m._conv_forward(input, weight, bias)


class QLinear(QModule):
    def _forward(
            self,
            input: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor
    ) -> torch.Tensor:
        """ Custom inference for quantized nn.Linear.
        Arguments:
            input (torch.Tensor): input tensor.
            weight (torch.Tensor): input weight.
            bias (torch.Tensor): input bias.

        Returns:
            output (torch.Tensor): result of quantized operation.
        """
        return F.linear(input, weight, bias)
