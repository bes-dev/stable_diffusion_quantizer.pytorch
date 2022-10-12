import torch
import torch.nn as nn


class Calibrator(nn.Module):
    """ Range calibrator.
    Arguments:
        momentum (float): the value used for the running average computation. Default: 0.1.
    """
    def __init__(self, m, momentum: float = 0.1) -> None:
        super().__init__()
        self.m = m
        self.momentum = momentum
        self.params = {}

    def mtype(self):
        return type(self.m)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward path.
        Arguments:
            input (torch.Tensor): input tensor.

        Returns:
            input (torch.Tensor): input tensor.
        """
        self._update_var(input.min().item(), "min")
        self._update_var(input.max().item(), "max")
        return self.m(input)

    def _smooth(self, observation: float, mixture: float) -> float:
        """ Smooth value using momentum
        Arguments:
            observation (float): new value.
            mixture (float): smoothed historical value.

        Returns:
            output (float): new smoothed value.
        """
        return self.momentum * observation + (1 - self.momentum) * mixture

    def _update_var(self, val: float, name: str) -> None:
        """ Update running average value by name.
        Arguments:
            val (float): new value.
            name (str): name of value.
        """
        if name in self.params:
            self.params[name] = self._smooth(val, self.params[name])
        else:
            self.params[name] = val
