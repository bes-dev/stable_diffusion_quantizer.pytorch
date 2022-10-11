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
import typing
import torch


class straight_through_estimator(torch.autograd.Function):
    """ Implementation of Straight Through Estimator."""
    @staticmethod
    def forward(ctx, input: torch.Tensor, op: typing.Callable) -> torch.Tensor:
        """ Forward path.
        Arguments:
            input (torch.Tensor): input tensor.
            op (typing.Callable): the operation that applies to input tensor.
                                  The operation should have a similar shape of the input and output.

        Returns:
            output (torch.Tensor): output tensor.
        """
        output = op(input)
        assert input.size() == output.size(), f"Shape mismatch: {input.size()} != {output.size()}."
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> typing.Tuple[torch.Tensor]:
        """ Backward path.
        Arguments:
            grad_output (torch.Tensor): output gradients.

        Returns:
            grad_input (typing.Tuple[torch.Tensor]): gradients for each input.
        """
        return grad_output, None


def logit(input: torch.Tensor) -> torch.Tensor:
    """Returns logit distribution."""
    return torch.log(input / (1 - input))


def binary_threshold(input: torch.Tensor, threshold: float = 0.5):
    """Binary threshold of the input tensor."""
    return (input > threshold).type(input.dtype)
