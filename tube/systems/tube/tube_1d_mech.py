# Copyright (c) 2025, twiinIT - All Rights Reserved
# twiinIT proprietary - See licence file packaged with this code

import numpy as np
from cosapp.systems import System
from scipy.interpolate import interp1d


class Tube1DMech(System):
    """A mechanical tube model.

    Computes the geometry of a tube under internal pressure.
    """

    def setup(self, connection_size=4):
        # Geometry (Initial and Deformed)
        self.add_inward("geom_cold", lambda s: 0.1)  # Initial cross-sectional area (m²)
        self.add_outward("geom", lambda s: 0.1)  # Deformed cross-sectional area (m²)

        # Mechanical properties
        self.add_inward("E", 0.1e9, unit="pa")  # Young's modulus (Pa)
        self.add_inward("thickness", 1.0e-3, unit="m")  # Thickness (converted to meters)

        # Aero (Pressure field)
        s_values = np.linspace(0, 1, connection_size)
        ps_values = np.full(connection_size, 101325.0)

        self.add_inward("Ps", np.column_stack((s_values, ps_values)), unit="pa")

    def compute(self):
        """Compute the deformed geometry based on pressure."""

        def deformed_area(s):
            """Compute the deformed cross-sectional area at a given position s."""
            # Initial area and radius
            A_cold = self.geom_cold(s)  # Initial cross-sectional area (m²)
            R_cold = np.sqrt(A_cold / np.pi)  # Compute initial radius (m)

            # Pressure at position s
            P = interp1d(self.Ps[:, 0], self.Ps[:, 1], kind="linear", fill_value="extrapolate")(s)

            # Tube material properties
            E = self.E  # Young's modulus (Pa)
            t = self.thickness  # Thickness (m)

            # Compute radial deformation
            delta_R = (P * R_cold) / (E * t)

            # Compute new radius and new area
            R_new = R_cold + delta_R  # New deformed radius
            A_new = np.pi * R_new**2  # New cross-sectional area

            return A_new  # Return updated area

        # Update the deformed geometry function
        self.geom = lambda s: deformed_area(s)
