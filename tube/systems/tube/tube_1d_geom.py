# Copyright (c) 2025, twiinIT - All Rights Reserved
# twiinIT proprietary - See licence file packaged with this code

import numpy as np
from cosapp.systems import System


class Tube1DGeom(System):
    """A geom tube model.

    Inputs
    ------
    d_in[m]: float
        inlet diameter
    d_exit[m]: float
        exit diameter
    length[m]: float
        tube length
    Outputs
    -------
    area_in[m**2]: float
        inlet area flow section
    area_exit[m**2]: float
        exit area flow section
    """

    def setup(self):
        # inwards
        self.add_inward("d_in", 0.1, unit="m")
        self.add_inward("d_exit", 0.1, unit="m")
        self.add_inward("length", 1.0, unit="m")

        # aero
        self.add_outward("geom", np.empty((0, 2)))

    def compute(self):
        self.geom = np.array(
                [
                    [0.0, np.pi * (self.d_in/2) ** 2],
                    [self.length, np.pi * (self.d_exit/2) ** 2],
                ]
            )
