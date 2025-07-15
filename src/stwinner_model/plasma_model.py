from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn
from torchdiffeq import odeint


class PlasmaModel(nn.Module):
    def __init__(self, x_width, y_width, h_param):
        super().__init__()
        self.h_param = h_param
        if h_param["architecture"] == "NeuralODE":
            self.model = NeuralODE(
                activation=nn.GELU,
                in_feature=x_width,
                nlayer=h_param["nlayer"],
                hidden=h_param["hidden"],
                out_feature=y_width,
            )
        else:
            self.model = None

    def forward(self, x):
        return self.model(x)


class NeuralODE(nn.Module):
    def __init__(self, activation, in_feature, nlayer, hidden, out_feature):
        super().__init__()
        self.activation = activation()
        linear_in = nn.Linear(in_feature, hidden)
        layers = OrderedDict([("linear_in", linear_in), ("act_in", self.activation)])
        self.encoder = nn.Sequential(layers)
        self.ode_block = ODEBlock(hidden, activation, nlayer)
        self.decoder = nn.Linear(hidden, out_feature)

    def forward(self, x):
        x = self.encoder(x)
        x = self.ode_block(x)
        x = self.decoder(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, hidden, activation, nlayer):
        super().__init__()
        self.odefunc = ODEfunc(hidden, activation, nlayer)
        self.integratioin_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integratioin_time = self.integratioin_time.type_as(x)
        out = odeint(self.odefunc, x, self.integratioin_time, method="rk4")
        return out[1]


class ODEfunc(nn.Module):
    def __init__(self, hidden, activation, nlayer):
        super().__init__()
        self.activation = activation()
        hidden_layer = LinearT(hidden, hidden)
        layers = []
        for i in range(nlayer - 1):
            layers.append(deepcopy(hidden_layer))
        self.layers = nn.ModuleList(layers)

    def forward(self, t, x):
        for l in self.layers:
            x = l(t, x)
            x = self.activation(x)
        return x


class LinearT(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], dim=1)
        return self._layer(ttx)
