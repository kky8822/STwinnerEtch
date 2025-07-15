from copy import deepcopy

import numpy as np
import torch
from torch import nn

from src.stwinner_model.etch_model import EtchModel
from src.stwinner_model.plasma_model import PlasmaModel


class STwinnerEtchModel(nn.Module):
    def __init__(self, h_param):
        super().__init__()
        self.h_param = h_param
        self.prof_width = self.h_param["etch_Etch"]["n_grid"] * self.h_param["etch_Etch"]["feature"]

    def build(self, cols_x, cols_y):
        model = nn.ModuleDict()
        for key, value in self.h_param.items():
            if value is not None:
                if "plasma" in key:
                    plasma_model = PlasmaModel(
                        x_width=len(cols_x[key]),
                        y_width=len(cols_y[key]),
                        h_param=self.h_param[key],
                    )
                    model[key] = deepcopy(plasma_model)
                elif "etch" in key:
                    etch_model = EtchModel(h_param=self.h_param[key])
                    model[key] = deepcopy(etch_model)
        self.model = model

    def plasma_block(self, key, rcp):
        flux = self.model[key](rcp)
        return flux

    def etch_block(self, key, prof_in, flux, etch_time):
        input = torch.concat((prof_in, flux, etch_time), dim=1)
        out = self.model[key](input)
        return out

    def forward(self, x):
        lc = np.asarray([
            self.prof_width,
            9,
            9,
            9,
            1,
            1,
            1,
        ])

        prof_in = x[:, :lc[:1].sum()]
        rcp_1 = x[:, lc[:1].sum(): lc[:2].sum()]
        rcp_2 = x[:, lc[:2].sum(): lc[:3].sum()]
        rcp_3 = x[:, lc[:3].sum(): lc[:4].sum()]
        etch_time_1 = x[:, lc[:4].sum(): lc[:5].sum()]
        etch_time_2 = x[:, lc[:5].sum(): lc[:6].sum()]
        etch_time_3 = x[:, lc[:6].sum(): lc[:7].sum()]

        # step 1
        flux_1 = self.plasma_block(key="plasma_Plasma", rcp=rcp_1)
        out_1 = self.etch_block(key="etch_Etch", prof_in = prof_in, flux=flux_1, etch_time=etch_time_1)

        # step 2
        flux_2 = self.plasma_block(key="plasma_Plasma", rcp=rcp_2)
        out_2 = self.etch_block(key="etch_Etch", prof_in=out_1[:, :self.prof_width], flux=flux_2, etch_time=etch_time_2)

        # step 3
        flux_3 = self.plasma_block(key="plasma_Plasma", rcp=rcp_3)
        out_3 = self.etch_block(key="etch_Etch", prof_in=out_2[:, :self.prof_width], flux=flux_3, etch_time=etch_time_3)

        out_mts = torch.concat((out_1[:, self.prof_width:], out_2[:, self.prof_width:], out_3[:, self.prof_width:]), dim=-1)
        out_prof = torch.concat((out_1[:, :self.prof_width], out_2[:, :self.prof_width], out_3[:, :self.prof_width]), dim=-1)
        out_flux = torch.concat((flux_1, flux_2, flux_3), dim=-1)

        out = torch.concat((out_mts, out_prof, out_flux), dim=-1)

        return out
