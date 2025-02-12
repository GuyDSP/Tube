# Copyright (c) 2025, twiinIT - All Rights Reserved
# twiinIT proprietary - See licence file packaged with this code

import numpy as np
from cosapp.systems import System
from scipy.interpolate import interp1d


class Tube1DGeom(System):
    """A geom tube model.

    Inputs
    ------
    Outputs
    -------
    """

    def setup(self):
        # inwards
        self.add_inward("d_in", 0.1, unit="m")
        self.add_inward("d_exit", 0.1, unit="m")
        self.add_inward("length", 1.0, unit="m")

        # aero
        self.add_outward("section", lambda s: 0.1)

    def compute(self):
        points = np.array(
            [
                np.pi * (self.d_in / 2) ** 2,
                np.pi * (self.d_exit / 2) ** 2,
            ]
        )

        self.section = lambda s: interp1d(
            [0.0, 1.0], points, kind="linear", fill_value="extrapolate"
        )(s)
