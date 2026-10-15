"""Tacotron-2 style PostNet. Not in the FastSpeech 2 paper, but used by the
reference implementations (ming024, xcmyz); configurable via model.postnet."""
import torch
import torch.nn as nn


class PostNet(nn.Module):
    def __init__(self, n_mels: int = 80, channels: int = 512, n_layers: int = 5, kernel: int = 5,
                 dropout: float = 0.5):
        super().__init__()
        pad = (kernel - 1) // 2
        layers = []
        for i in range(n_layers):
            cin = n_mels if i == 0 else channels
            cout = n_mels if i == n_layers - 1 else channels
            block = [nn.Conv1d(cin, cout, kernel, padding=pad), nn.BatchNorm1d(cout)]
            if i < n_layers - 1:
                block.append(nn.Tanh())
            block.append(nn.Dropout(dropout))
            layers.append(nn.Sequential(*block))
        self.convs = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        keep = None if pad_mask is None else (~pad_mask).unsqueeze(1).to(x.dtype)       # (B,1,T)
        y = x.transpose(1, 2)
        for c in self.convs:
            if keep is not None:
                y = y * keep                 # zero padded frames between layers (batch invariance)
            y = c(y)
        if keep is not None:
            y = y * keep
        return y.transpose(1, 2)
