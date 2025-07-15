from collections import OrderedDict
from copy import deepcopy
from functools import partial

import torch
from torch import nn
from torch.nn import ModuleList
import torch.nn.functional as F


class EtchModel(nn.Module):
    def __init__(self, h_param):
        super().__init__()
        self.h_param = h_param
        architecture = self.h_param["architecture"]

        if architecture == "FNO":
            self.model = FNO(
                n_grid=h_param["n_grid"],
                in_feature=h_param["feature"] + 1,
                out_feature=h_param["feature"],
                input_width=h_param["input_width"],
                modes=h_param["modes"],
                nlayer=h_param["nlayer"],
                hidden=h_param["hidden"],
                mts_idx=h_param["mts_idx"],
            )
        else:
            self.model = None

    def forward(self, x):
        x = self.model(x)
        return x


class FNO(nn.Module):
    def __init__(
        self,
        n_grid,
        in_feature,
        out_feature,
        input_width,
        modes,
        nlayer,
        hidden,
        mts_idx,
    ):
        super().__init__()
        self.modes = modes
        self.width = hidden
        self.padding = 3
        self.n_grid = n_grid
        self.input_width = input_width
        self.mts_idx = mts_idx
        self.nlayer = nlayer
        self.activation = nn.GELU()

        self.param_encoder = nn.Sequential(
            nn.Linear(self.input_width + 1, (self.n_grid + self.padding) * self.width),
            Reshape(self.n_grid + self.padding, self.width),
        )
        self.encoder = Encoder(in_feature, hidden)
        self.decoder = Decoder(hidden, out_feature)

        operator_block = FNOBlock(self.width + self.width, self.width, self.modes)
        self.layers = ModuleList([deepcopy(operator_block) for _ in range(nlayer)])

    def forward(self, x):
        b, _ = x.shape
        profile_in = x[:, : self.n_grid].reshape(-1, 1, self.n_grid).permute(0, 2, 1)
        input = x[:, self.n_grid : self.n_grid + self.input_width]
        time = x[:, self.n_grid + self.input_width :]

        b, n, _ = profile_in.shape
        param = torch.concat((input, time), dim=-1)

        l = self.encoder(profile_in)
        param = self.param_encoder(param)

        l = l.permute(0, 2, 1)
        param = param.permute(0, 2, 1)

        l = F.pad(l, [0, self.padding])
        for i in range(self.nlayer):
            l = torch.concat((l, param), dim=1)
            l = self.layers[i](l)
        l = l[..., : -self.padding]

        l = l.permute(0, 2, 1)
        profile_out = self.decoder(l)
        profile_out = profile_out.permute(0, 2, 1)

        mts = profile_out[:, 0, self.mts_idx]

        out = torch.concat((profile_out.reshape(b, -1), mts), dim=-1)

        return out


class FNOBlock(nn.Module):
    def __init__(self, in_feature, out_feature, modes):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.modes = modes

        self.conv = SpectralConv1d(self.in_feature, self.out_feature, self.modes)
        self.w = nn.Conv1d(self.in_feature, self.out_feature, 1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = F.gelu(x)
        return x


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, 2)
        )

    def compl_mul1d(self, a, b):
        op = partial(torch.einsum, "bix,iox->box")
        return torch.stack(
            [
                op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
                op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1]),
            ],
            dim=-1,
        )

    def forward(self, x):
        b = x.shape[0]
        x_ft = torch.fft.rfft(x)
        x_ft = torch.view_as_real(x_ft)

        out_ft = torch.zeros(
            b, self.out_channels, x.size(-1) // 2 + 1, 2, device=x.device
        )
        out_ft[:, :, : self.modes] = self.compl_mul1d(
            x_ft[:, :, : self.modes], self.weight
        )

        out_ft = torch.view_as_complex(out_ft)
        x = torch.fft.irfft(out_ft, n=x.size(-1))

        return x


class Reshape(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.n = n
        self.m = m

    def forward(self, x):
        x = x.reshape(-1, self.n, self.m)
        return x


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        x = x.permute(self.dims[0], self.dims[1], self.dims[2])
        return x


class ResNet(nn.Module):
    def __init__(self, out_feature, kernel, slide, padding, padding_mode):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(
            out_feature,
            out_feature,
            kernel,
            slide,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.act = nn.GELU()
        self.conv1d_2 = nn.Conv1d(
            out_feature,
            out_feature,
            kernel,
            slide,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        x = self.conv1d_2(self.act(self.conv1d_1(x))) + x
        return x


class Encoder(nn.Module):
    def __init__(self, in_feature, width):
        super().__init__()

        encoder_dict = []
        encoder_dict.append(("linear", nn.Linear(in_feature, width)))
        encoder_dict.append(("permute", Permute(0, 2, 1)))
        encoder_dict.append(
            ("ResNet", ResNet(width, 3, 1, padding=1, padding_mode="replicate"))
        )
        encoder_dict.append(("perumte2", Permute(0, 2, 1)))
        self.encoder = nn.Sequential(OrderedDict(encoder_dict))

    def forward(self, x):
        b, n, c = x.shape
        x = torch.concat(
            (x, torch.linspace(0, 1, n).unsqueeze(0).unsqueeze(-1).tile(b, 1, 1).to(x)),
            dim=-1,
        )
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, width, out_feature):
        super().__init__()

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_feature)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc2(self.activation(self.fc1(x)))
        return x
