# Copyright (C) 2025, twiinIT
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from cosapp.drivers import NonLinearSolver
from pyturbo.thermo import IdealDryAir
from scipy.optimize import fsolve

from tube.systems.tube import Tube1D, Tube1DAero, Tube1DGeom, Tube1DMech


@pytest.fixture
def aero_system():
    """Fixture to create an instance of Tube1DAero."""
    return Tube1DAero("tube")


@pytest.fixture
def gas():
    """Fixture to create an instance of IdealDryAir."""
    return IdealDryAir()


class TestBasic:
    """Define tests for the structure model."""

    def test_dim(self, aero_system):
        """Test shape consistency for q_from_wpt and wpt_from_q."""
        n = 10
        w, pt, tt, area = [np.full((n), v) for v in (10.0, 100000.0, 300.0, 0.1)]

        q = aero_system.q_from_wpt(w, pt, tt, area, subsonic=True)

        assert q.shape == (n, 3)
        assert aero_system.wpt_from_q(q, area)[0].shape == (n,)

    def test_wpt_conversion(self, aero_system, gas):
        """Test the consistency of wpt_from_q and q_from_wpt."""
        area, mach, ps, ts = 0.1, 0.5, 100000.0, 300.0
        density, tt, pt, c = self._compute_gas_properties(gas, ps, ts, mach)

        u = mach * c
        w = density * u * area
        E = gas.h(ts) + 0.5 * u * u - ps / density

        q_exact = density * area * np.array([1.0, u, E])
        q_sim = aero_system.q_from_wpt(w, pt, tt, area, subsonic=True)

        np.testing.assert_allclose(q_sim, q_exact, rtol=1e-6)
        w_sim, p_sim, t_sim = aero_system.wpt_from_q(q_sim, area)

        assert pytest.approx(w_sim, 1e-6) == w
        assert pytest.approx(p_sim, 1e-6) == pt
        assert pytest.approx(t_sim, 1e-6) == tt

    def test_wpt_vector_conversion(self, aero_system, gas):
        """Test wpt_from_q and q_from_wpt using vectorized operations."""
        n = 10
        area, mach, ps, ts = [np.full((n), v) for v in (0.1, 0.5, 100000.0, 300.0)]
        density, tt, pt, c = self._compute_gas_properties(gas, ps, ts, mach)

        u = mach * c
        w = density * u * area
        E = gas.h(ts) + 0.5 * u * u - ps / density

        q_exact = np.column_stack([density * area, density * area * u, density * area * E])
        q_sim = aero_system.q_from_wpt(w, pt, tt, area, subsonic=True)

        np.testing.assert_allclose(q_sim, q_exact, rtol=1e-6)

        w_sim, p_sim, t_sim = aero_system.wpt_from_q(q_sim, area)

        assert pytest.approx(w_sim, 1e-6) == w
        assert pytest.approx(p_sim, 1e-6) == pt
        assert pytest.approx(t_sim, 1e-6) == tt

    def test_rupEc_from_q(self, aero_system, gas):
        """Test rupEc_from_q function."""
        area, mach, ps, ts = 0.1, 0.5, 100000.0, 300.0
        density, _, _, c = self._compute_gas_properties(gas, ps, ts, mach)

        u = mach * c
        E = gas.h(ts) + 0.5 * u * u - ps / density
        q_exact = density * area * np.array([1.0, u, E])

        sim = aero_system.rupEc_from_q(q_exact, area)

        expected_values = [density, u, ps, E, c]
        for i, expected in enumerate(expected_values):
            assert pytest.approx(sim[i], 1e-6) == expected

    @staticmethod
    def _compute_gas_properties(gas, ps, ts, mach):
        """Helper function to compute gas properties."""
        density = gas.density(ps, ts)
        tt = gas.total_t(ts, mach)
        pt = gas.total_p(ps, ts, tt)
        c = gas.c(ts)
        return density, tt, pt, c


class TestTube1DGeom:
    """Define tests for the Tube1DGeom model."""

    def test_run_once(self):
        Tube1DGeom("tube").run_once()


class TestTube1DAero:
    """Define tests for the Tube1DAero model."""

    def test_setup(self):
        Tube1DAero("tube")

    def test_run_once(self):
        sys = Tube1DAero("tube")
        sys.run_once()
        assert sys.res < sys.ftol


class TestTube1DMech:
    """Define tests for the Tube1DAero model."""

    def test_setup(self):
        Tube1DMech("tube")

    def test_run_once(self):
        sys = Tube1DMech("tube")
        sys.run_once()


class TestTube1D:
    """Define tests for the Tube1D model."""

    def test_setup(self):
        Tube1D("tube")

    def test_run_once(self):
        sys = Tube1D("tube")
        sys.run_once()
        assert sys.aero.res < sys.aero.ftol

    def test_run_driver(self):
        sys = Tube1D("tube")
        solver = sys.add_driver(NonLinearSolver("solver", max_iter=2))
        sys.run_drivers()
        residu = np.linalg.norm(solver.problem.residue_vector())

        print(solver.problem)

        assert sys.aero.res < sys.aero.ftol
        assert residu < 1e-6

    def test_run_once_uniform(self):
        """Test Tube1D in uniform conditions."""
        sys = Tube1D("tube")
        area_in, area_exit = 0.1, 0.1
        sys.d_in = np.sqrt(area_in / np.pi) * 2
        sys.d_exit = np.sqrt(area_exit / np.pi) * 2

        # Numerical solution
        sys.fl_in.W = 1.0
        sys.fl_in.Pt, sys.fl_in.Tt = 100000.0, 300.0

        sys.run_once()

        assert sys.aero.res < sys.aero.ftol
        assert sys.aero.it == 1

    def test_run_once_subsonic(self):
        """Test Tube1D in subsonic conditions."""
        sys = Tube1D("tube")
        gas = IdealDryAir()
        Pt_in, Tt_in, ps_exit = 100000.0, 300.0, 80000.0
        area_in, area_exit = 0.1, 0.1

        sys.d_in = np.sqrt(area_in / np.pi) * 2
        sys.d_exit = np.sqrt(area_exit / np.pi) * 2

        def mass_flow(W):
            return ps_exit - gas.static_p(
                Pt_in, Tt_in, gas.mach(Pt_in, Tt_in, W[0] / area_exit, subsonic=True)
            )

        W_in = fsolve(mass_flow, x0=[1.0])[0]

        # Numerical solution
        sys.fl_in.W = W_in * 1.01
        sys.fl_in.Pt, sys.fl_in.Tt = Pt_in, Tt_in
        sys.run_once()

        assert sys.aero.res < sys.aero.ftol
        assert pytest.approx(sys.fl_out.Pt, rel=1e-3) == Pt_in
        assert pytest.approx(sys.fl_out.Tt, rel=1e-3) == Tt_in
        assert pytest.approx(sys.fl_out.W, rel=2e-2) == W_in
        assert sys.aero.Ps[1, -1] > ps_exit
