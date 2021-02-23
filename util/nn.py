import torch
import torch.nn as nn

from typing import Sequence, Optional

class FeedForward(nn.Module):
    def __init__(self, in_size: int,
                 layer_sizes: Sequence[int],
                 out_size: Optional[int] = None # there is no activation after `out_size`
                 ) -> None:

        super(FeedForward, self).__init__()

        x = in_size
        for i, layer_size in enumerate(layer_sizes):
            linear_layer = nn.Linear(x, layer_size, bias=True)
            activation_layer = nn.ELU()
            self._layers.append(linear_layer)
            self._layers.append(activation_layer)
            x = layer_size

        if out_size is not None:
            self._layers.append(
                nn.Linear(x, out_size, bias=True)
            )

        self.sequential = nn.Sequential(*self._layers)

    def forward(self, x: torch.Tensor):
        out = self.sequential(x)
        return out
