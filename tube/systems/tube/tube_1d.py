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

    def setup(self, connection_size=4):
        # Physics
        self.add_child(Tube1DGeom("geom"), pulling=["d_in", "d_exit", "length"])
        self.add_child(Tube1DMech("mech", connection_size=connection_size))
        self.add_child(
            Tube1DAero("aero", connection_size=connection_size), pulling=["fl_in", "fl_out"]
        )

        # connections
        self.connect(self.geom.outwards, self.mech.inwards, {"section": "section_cold"})
        self.connect(self.mech.outwards, self.aero.inwards, ["section"])
        self.connect(self.aero.outwards, self.mech.inwards, ["Ps"])
