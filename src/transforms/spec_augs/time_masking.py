from torch import Tensor, nn
from torchaudio import transforms as T


class TimeMasking(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = T.TimeMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        for i in range(4):
            x = self._aug(x)
        return x.squeeze(1)