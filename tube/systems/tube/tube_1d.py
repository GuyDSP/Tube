# Copyright (c) 2023, twiinIT - All Rights Reserved
# twiinIT proprietary - See licence file packaged with this code

from cosapp.systems import System

from tube.systems.tube import Tube1DAero, Tube1DGeom, Tube1DMech


class Tube1D(System):
    """tube model.

    Inputs
    ------
    fl_in: FluidPort
        inlet fluid

    Outputs
    -------
    fl_out: FluidPort
        exit fluid
    """

    def setup(self):
        # Physics
        self.add_child(Tube1DGeom("geom"), pulling=["d_in", "d_exit", "length"])
        self.add_child(Tube1DMech("mech"))
        self.add_child(Tube1DAero("aero"), pulling=["fl_in", "fl_out"])

        # connections
        self.connect(self.geom.outwards, self.mech.inwards, {"geom": "geom_cold"})
        self.connect(self.mech.outwards, self.aero.inwards, ["geom"])
