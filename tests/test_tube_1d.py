# Copyright (C) 2022-2023, twiinIT
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from pyturbo.thermo import IdealDryAir
from scipy.optimize import fsolve

from tube.systems.tube import Tube1D, Tube1DAero


class TestBasic:
    """Define tests for the structure model."""

    def test_dim(self):
        sys = Tube1DAero("tube")
        n = 10

        w = np.full((n), 10.0)
        pt = np.full((n), 100000.0)
        tt = np.full((n), 300.0)
        area = np.full((n), 0.1)

        q = sys.q_from_wpt(w, pt, tt, area, subsonic=True)

        assert q.shape == (n, 3)
        assert sys.wpt_from_q(q, area)[0].shape == (n,)

    def test_wpt_from_q_from_wpt(self):
        sys = Tube1DAero("tube")
        gas = IdealDryAir()

        area = 0.1
        mach = 0.5
        ps = 100000.0
        ts = 300.0

        density = gas.density(ps, ts)
        tt = gas.total_t(ts, mach)
        pt = gas.total_p(ps, ts, tt)
        c = gas.c(ts)

        u = mach * c
        w = density * u * area
        E = gas.h(ts) + 0.5 * u * u - ps / density

        q_exact = density * area * np.array([1.0, u, E])

        q_sim = sys.q_from_wpt(w, pt, tt, area, subsonic=True)

        np.testing.assert_allclose(q_sim, q_exact, rtol=1e-6)

        w_sim, p_sim, t_sim = sys.wpt_from_q(q_sim, area)

        assert pytest.approx(w_sim, 1e-6) == w
        assert pytest.approx(p_sim, 1e-6) == pt
        assert pytest.approx(t_sim, 1e-6) == tt

    def test_wpt_from_q_from_wpt_vector(self):
        sys = Tube1DAero("tube")
        gas = IdealDryAir()

        n = 10

        area = np.full((n), 0.1)
        mach = np.full((n), 0.5)
        ps = np.full((n), 100000.0)
        ts = np.full((n), 300.0)

        density = gas.density(ps, ts)
        tt = gas.total_t(ts, mach)
        pt = gas.total_p(ps, ts, tt)
        c = gas.c(ts)

        u = mach * c
        w = density * u * area
        E = gas.h(ts) + 0.5 * u * u - ps / density

        q_exact = np.transpose(np.array([density * area, density * area * u, density * area * E]))

        q_sim = sys.q_from_wpt(w, pt, tt, area, subsonic=True)

        np.testing.assert_allclose(q_sim, q_exact, rtol=1e-6)

        w_sim, p_sim, t_sim = sys.wpt_from_q(q_sim, area)

        assert pytest.approx(w_sim, 1e-6) == w
        assert pytest.approx(p_sim, 1e-6) == pt
        assert pytest.approx(t_sim, 1e-6) == tt

    def test_rupEc_from_q(self):
        sys = Tube1DAero("tube")
        gas = IdealDryAir()

        area = 0.1
        mach = 0.5
        ps = 100000.0
        ts = 300.0
        density = gas.density(ps, ts)
        c = gas.c(ts)

        u = mach * c
        E = gas.h(ts) + 0.5 * u * u - ps / density

        q_exact = density * area * np.array([1.0, u, E])

        sim = sys.rupEc_from_q(q_exact, area)

        assert pytest.approx(sim[0], 1e-6) == density
        assert pytest.approx(sim[1], 1e-6) == u
        assert pytest.approx(sim[2], 1e-6) == ps
        assert pytest.approx(sim[3], 1e-6) == E
        assert pytest.approx(sim[4], 1e-6) == c


class TestTube1D:
    """Define tests for the structure model."""

    def test_run_once_uniform(self):
        sys = Tube1D("tube")
        area_in = np.pi * (sys.d_in / 2) ** 2

        # numerical solution
        sys.fl_in.W = 1.0
        mach = IdealDryAir().mach(sys.fl_in.Pt, sys.fl_in.Tt, sys.fl_in.W / area_in, subsonic=True)
        sys.Ps_out = IdealDryAir().static_p(sys.fl_in.Pt, sys.fl_in.Tt, mach)

        sys.run_once()

        assert sys.aero.res < sys.aero.ftol

    def test_run_once_open(self):
        sys = Tube1D("tube")
        area_in = np.pi * (sys.d_in / 2) ** 2
        sys.d_exit = sys.d_in * 1.1

        # numerical solution
        sys.fl_in.W = 1.0
        mach = IdealDryAir().mach(sys.fl_in.Pt, sys.fl_in.Tt, sys.fl_in.W / area_in, subsonic=True)
        sys.Ps_out = IdealDryAir().static_p(sys.fl_in.Pt, sys.fl_in.Tt, mach)

        sys.run_once()

        assert sys.aero.res < sys.aero.ftol


class TestTube1DAero:
    """Define tests for the structure model."""

    def test_run_once_uniform(self):
        sys = Tube1DAero("tube")

        area_in = 0.1
        area_exit = 0.1
        sys.geom = np.array([[0.0, area_in], [1.0, area_exit]])

        # numerical solution
        sys.fl_in.W = 1.0
        mach = IdealDryAir().mach(sys.fl_in.Pt, sys.fl_in.Tt, sys.fl_in.W / area_in, subsonic=True)
        sys.Ps_out = IdealDryAir().static_p(sys.fl_in.Pt, sys.fl_in.Tt, mach)

        sys.scheme = "LW"
        sys.implicit = False
        sys.run_once()

        assert sys.res < sys.ftol
        assert sys.it == 1

    def test_run_once_subsonic(self):
        sys = Tube1DAero("tube")

        # exact solution
        Pt_in = 100000.0
        Tt_in = 300.0

        area_in = 0.1
        area_exit = 0.11
        sys.geom = np.array([[0.0, area_in], [1.0, area_exit]])

        Ps_out = 80000.0

        gas = IdealDryAir()

        def func(W):
            mach_out = gas.mach(Pt_in, Tt_in, W[0] / area_exit, subsonic=True)
            return Ps_out - gas.static_p(
                Pt_in,
                Tt_in,
                mach_out,
            )

        W_in = fsolve(func, x0=[1.0])[0]

        # numerical solution
        sys.fl_in.W = W_in * 1.01

        sys.fl_in.Pt = Pt_in
        sys.fl_in.Tt = Tt_in
        sys.Ps_out = Ps_out
        sys.ftol = 1e-3
        sys.run_once()

        assert sys.res < sys.ftol
        assert pytest.approx(sys.fl_out.Pt, rel=1e-3) == Pt_in
        assert pytest.approx(sys.fl_out.Tt, rel=1e-3) == Tt_in
        assert pytest.approx(sys.fl_out.W, rel=2e-2) == W_in
        assert pytest.approx(sys.get_Ps()[-1], rel=1e-2) == Ps_out

    def test_run_once_supersonic(self):
        sys = Tube1DAero("tube")

        # exact solution
        Pt_in = 100000.0
        Tt_in = 300.0
        W_in = 10.0

        area_in = 0.1
        area_exit = 0.2
        sys.geom = np.array([[0.0, area_in], [1.0, area_exit]])

        gas = IdealDryAir()
        mach_in = gas.mach(Pt_in, Tt_in, W_in / area_in, subsonic=False)
        mach_exit = gas.mach(Pt_in, Tt_in, W_in / area_exit, subsonic=False)

        # numerical solution
        sys.fl_in.W = W_in
        sys.fl_in.Pt = Pt_in
        sys.fl_in.Tt = Tt_in
        sys.subsonic = False

        sys.ftol = 1e-3
        sys.run_once()

        assert sys.res < sys.ftol
        assert pytest.approx(sys.get_mach()[0], rel=1e-2) == mach_in
        assert pytest.approx(sys.get_mach()[-1], rel=1e-1) == mach_exit

        # implicit
        sys.implicit = True
        sys.CFL = 5.0
        sys.run_once()

        assert sys.res < sys.ftol
        assert pytest.approx(sys.get_mach()[0], rel=1e-2) == mach_in
        assert pytest.approx(sys.get_mach()[-1], rel=1e-1) == mach_exit

    def test_run_once_supersonic_LW(self):
        sys = Tube1DAero("tube")

        # exact solution
        Pt_in = 100000.0
        Tt_in = 300.0
        W_in = 10.0

        area_in = 0.1
        area_exit = 0.2
        sys.geom = np.array([[0.0, area_in], [1.0, area_exit]])

        gas = IdealDryAir()
        mach_in = gas.mach(Pt_in, Tt_in, W_in / area_in, subsonic=False)
        mach_exit = gas.mach(Pt_in, Tt_in, W_in / area_exit, subsonic=False)

        # numerical solution
        sys.fl_in.W = W_in
        sys.fl_in.Pt = Pt_in
        sys.fl_in.Tt = Tt_in
        sys.subsonic = False
        sys.scheme = "LW"

        sys.ftol = 1e-3

        sys.run_once()

        assert sys.res < sys.ftol
        assert pytest.approx(sys.get_mach()[0], rel=1e-2) == mach_in
        assert pytest.approx(sys.get_mach()[-1], rel=1e-1) == mach_exit
