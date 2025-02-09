# Copyright (C) 2022-2023, twiinIT
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from cosapp.drivers import NonLinearSolver

from tube.systems import Tube


class TestTube:
    """Define tests for the tube model."""

    def test_run_once(self):
        sys = Tube("sys")

        sys.fl_in.W = 10.0
        sys.run_drivers()

        assert sys.fl_out.W == sys.fl_in.W

    def test_run_nls(self):
        sys = Tube("sys")

        solver = sys.add_driver(NonLinearSolver("nls"))
        sys.run_drivers()
        residu = np.linalg.norm(solver.problem.residue_vector())

        assert residu < 1e-6
        assert sys.geom_hot.d_in > sys.geom_hot.d_exit > sys.geom.d_exit
