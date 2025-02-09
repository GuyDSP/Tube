# Copyright (c) 2023, twiinIT - All Rights Reserved
# twiinIT proprietary - See licence file packaged with this code

from cosapp.systems import System


class TubeMech(System):
    """A mechanical tube model.

    Inputs
    ------
    Outputs
    -------
    """

    def setup(self):
        # geom
        self.add_inward("d_in", 0.1, unit="m")
        self.add_inward("d_exit", 0.1, unit="m")
        self.add_inward("thickness", 0.1e-3, unit="m")

        self.add_outward("d_hot_in", 0.1, unit="m")
        self.add_outward("d_hot_exit", 0.1, unit="m")

        # mech
        self.add_inward("E", 0.1e9, unit="pa")
        self.add_outward("eps_r_in", 0.0, unit="")
        self.add_outward("eps_r_exit", 0.0, unit="")

        # aero
        self.add_inward("ps_in", 1e5, unit="pa")
        self.add_inward("ps_exit", 1e5, unit="pa")

    def compute(self):
        self.eps_r_in = self.ps_in * self.d_in / (4 * self.thickness * self.E)
        self.eps_r_exit = self.ps_exit * self.d_exit / (4 * self.thickness * self.E)

        self.d_hot_in = self.d_in * (1 + self.eps_r_in)
        self.d_hot_exit = self.d_exit * (1 + self.eps_r_exit)
