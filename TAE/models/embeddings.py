import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class FPEmbedding(nn.Module):

    def __init__(self, nvector, nemb, dropout=0.2):
        super().__init__()

        self.nvector = nvector  # size of the input vector to embed
        self.nemb = nemb  # embedding dimension
        self.dropout = nn.Dropout(p=dropout)

        self.linear_transform = nn.Linear(nvector, nemb)
        self.norm = nn.LayerNorm(nemb)

        self.init_weights()

    def init_weights(self):

        for layer in [self.linear_transform]:
            if isinstance(layer, nn.Linear):
                if hasattr(layer, "weight"):
                    nn.init.xavier_normal_(layer.weight)
                if hasattr(layer, "bias"):
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # input is assumed to be N_seq x N_batch x N_vec
        if x.shape[-1] != self.nvector:
            if self.nvector != 1:
                raise ValueError("Input Vector Has the incorrect shape!")
            return self.forward(torch.unsqueeze(x, -1))
        x = self.linear_transform(x)
        x = self.norm(x)
        return x
