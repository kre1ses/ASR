import random

from torch import Tensor, nn
from torchaudio import transforms as T


class TimeMasking(nn.Module):
    def __init__(self, p, *args, **kwargs):
        super().__init__()
        self.p = p
        self._aug = T.TimeMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        if random.random() < self.p:
            x = data.unsqueeze(1)
            return self._aug(x).squeeze(1)
        else:
            return data