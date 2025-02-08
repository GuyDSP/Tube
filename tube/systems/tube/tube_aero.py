# Copyright (c) 2023, twiinIT - All Rights Reserved
# twiinIT proprietary - See licence file packaged with this code

import numpy as np
from cosapp.systems import System
from pyturbo.thermo import IdealDryAir
from pyturbo.ports import FluidPort


class TubeAero(System):
    """A aero tube model.

    Inputs
    ------
    fl_in: FluidPort
        inlet fluid
    area_in[m**2]: float
        inlet area flow section
    area_exit[m**2]: float
        exit area flow section
    k[]: float, default=0.1
        pressure loss coefficient

    Outputs
    -------
    fl_out: FluidPort
        exit fluid
    """

    def setup(self):
        # inputs/outputs
        self.add_input(FluidPort, "fl_in")
        self.add_output(FluidPort, "fl_out")

        # inwards
        self.add_inward('gas', IdealDryAir())
        self.add_inward("f", 0.1, unit="")

        # outwards
        self.add_outward("ps_in", 1e5, unit="pa")
        self.add_outward("ps_exit", 1e5, unit="pa")
        self.add_outward("mach_in", 0.0, unit="")
        self.add_outward("mach_exit", 0.0, unit="")

        # geometry
        self.add_inward("area_in", 0.1, unit="m**2")
        self.add_inward("area_exit", 0.1, unit="m**2")
        self.add_inward("length", 1.0, unit="m")

    def compute(self):
        # approx rho
        rho = self.gas.density(self.fl_in.Pt, self.fl_in.Tt)

        # nominal output computation from Darcy-Weisbach equation
        self.fl_out.W = self.fl_in.W
        self.fl_out.Tt = self.fl_in.Tt

        v_in = self.fl_in.W / (self.area_in * rho)
        d_in = np.sqrt(4 * self.area_in / np.pi)
        self.fl_out.Pt = self.fl_in.Pt - self.f * (self.length / d_in) * (rho * v_in**2 / 2)

        # static pressure
        self.ps_in = self.fl_in.Pt - 0.5 * rho * v_in**2

        v_exit = self.fl_out.W / (self.area_exit * rho)
        self.ps_exit = self.fl_out.Pt - 0.5 * rho * v_exit**2
