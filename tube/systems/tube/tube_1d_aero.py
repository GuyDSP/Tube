# Copyright (c) 2025, twiinIT - All Rights Reserved
# twiinIT proprietary - See licence file packaged with this code

import numpy as np
from cosapp.base import System
from pyturbo.ports import FluidPort
from pyturbo.thermo import IdealDryAir
from scipy.interpolate import interp1d


class Tube1DAero(System):
    """ """

    def __init__(self, name):
        self._area = None
        self._x = None
        self._q = None
        super().__init__(name)

    def setup(self):
        self.add_inward("gas", IdealDryAir())

        # inputs/outputs
        self.add_input(FluidPort, "fl_in")
        self.add_output(FluidPort, "fl_out")

        # geometry
        points = np.array(
            [
                [0.0, 0.1],
                [1.0, 0.1],
            ]
        )
        self.add_inward(
            "geom",
            lambda s: interp1d(points[:, 0], points[:, 1], kind="linear", fill_value="extrapolate")(
                s
            ),
        )

        # inwards
        self.add_inward("subsonic", True, desc="initial inlet flow status")

        # outwards
        self.add_outward("Ps", lambda s: 101325.0, desc="static pressure")

        # 1d solver
        self.add_inward("n", 11, desc="number of cells")
        self.add_inward("CFL", 2.0, desc="CFL number")
        self.add_inward("implicit", True)
        self.add_inward("scheme", "Roe")

        self.add_inward("ftol", 1e-6)
        self.add_inward("it_max", 10000)

        self.add_outward("res", 0.0)
        self.add_outward("it", 0)

    def compute(self):
        # For 1D modelisation : https://perso.univ-lyon1.fr/marc.buffat/COURS/AERO_HTML/node45.html
        # For scheme : https://encyclopediaofmath.org/wiki/Lax-Wendroff_method
        #              https://www.psvolpiani.com/courses

        # mesh
        s_mesh = np.linspace(0.0, 1.0, self.n)

        s_in = 2 * s_mesh[0] - s_mesh[1]
        s_exit = 2 * s_mesh[-1] - s_mesh[-2]
        self._x = np.concatenate(([s_in], s_mesh, [s_exit]))

        self._area = self.geom(s_mesh)

        # solver
        gas = self.gas

        n = self.n
        it = 0
        res = np.inf
        CFL = self.CFL

        # mesh
        area = self._area
        dx = (self._x - np.roll(self._x, 1))[1:-1]

        # init flow with input value
        if self._q is None:
            W = np.full((n), self.fl_in.W)
            Pt = np.full((n), self.fl_in.Pt)
            Tt = np.full((n), self.fl_in.Tt)

            q = self.q_from_wpt(W, Pt, Tt, area, subsonic=self.subsonic)
        else:
            q = self._q

        # implcit
        if self.implicit:
            A = (
                (1 + CFL**2) * np.diagflat(np.ones((n - 2)))
                - 0.5 * CFL**2 * np.diagflat(np.ones((n - 3)), 1)
                - 0.5 * CFL**2 * np.diagflat(np.ones((n - 3)), -1)
            )
            invA = np.linalg.inv(A)

        while res > self.ftol and it < self.it_max:
            # inlet boundary layer
            m = self.mach_from_q(q[1], area[1])

            if m >= 1:  # inlet supersonic, all imposed
                q[0] = self.q_from_fluid_port(self.fl_in, area[0], subsonic=False)
            elif 0 <= m < 1:  # inlet subsonic, pt, tt imposed, w extrapolated
                q[0] = self.q_from_wpt(
                    q[1, 1], self.fl_in.Pt, self.fl_in.Tt, area[0], subsonic=True
                )
            elif m < 0:  # inlet reverse flow
                q[0] = q[1]
            else:
                raise ValueError("negative flow at inlet")

            # outlet bondary layer
            m = self.mach_from_q(q[-2], area[-2])

            if m >= 1:  # outlet supersonic, all extrapolated
                q[-1] = q[-2]
            elif 1 > m >= 0:  # outlet subsonic, ps imposed, pt and tt extrapolated
                _, pt, tt = self.wpt_from_q(q[-2], area[-2])
                ps = gas.static_p(pt, tt, m)

                mach = gas.mach_ptpstt(pt, ps, tt)
                ts = gas.static_t(tt, mach)
                c = gas.c(ts)
                density = gas.density(ps, ts)
                w = area[-2] * density * mach * c

                qn = self.q_from_wpt(w, pt, tt, area[-1], subsonic=True)
                q[-1] = qn
            else:
                raise ValueError("negative speed at exit")

            # schemes in conservative form
            _, u, p, _, c = self.rupEc_from_q(q, area)
            dt = CFL * dx / (abs(u) + c)
            if self.scheme == "LW":
                f12 = self.flux_LW(q, dt, dx, area)
            elif self.scheme == "Roe":
                f12 = self.flux_roe(q, area)
            else:
                raise ValueError(f"Scheme '{self.scheme}' unknown")

            df = f12[:-1] - f12[1:]

            # sources : wall boundarylayer
            df[:, 1] -= p[1:-1] * (np.roll(area, 1) - area)[1:-1]

            # implicit
            if self.implicit:
                df = np.matmul(invA, df)

            # update
            res = np.max(abs(df))
            q[1:-1] = q[1:-1] + (df.T * dt[1:-1] / dx[1:-1]).T

            it += 1

        # primary variable

        self.fl_out.W, self.fl_out.Pt, self.fl_out.Tt = self.wpt_from_q(q[-2], area[-2])

        self.res = res
        self.it = it

        self._q = q

        _, _, Ps, _, _ = self.rupEc_from_q(self._q, self._area)
        self.Ps = lambda s: interp1d(self._x[1:-1], Ps, kind="linear", fill_value="extrapolate")(s)

    def get_u(self):
        _, u, _, _, _ = self.rupEc_from_q(self._q, self._area)
        return u

    def get_mach(self):
        _, u, _, _, c = self.rupEc_from_q(self._q, self._area)
        return u / c

    def flux_LW(self, q, dt, dx, s):
        # flux at cell center
        f = self.flux_from_q(q, s)

        # flux at cell boundaries
        dt12 = 1 / 2 * (dt[:-1] + dt[1:])
        dx12 = 1 / 2 * (dx[:-1] + dx[1:])
        q12 = 1 / 2 * (q[:-1] + q[1:]) + 1 / 2 * ((f[:-1] - f[1:]).T * dt12 / dx12).T
        area12 = 1 / 2 * (s[:-1] + s[1:])

        return self.flux_from_q(q12, area12)

    def flux_roe(self, q, s):
        gamma = self.gas.gamma(0)
        nx = q.shape[0]

        # Compute Roe averages

        q12 = 1 / 2 * (q[:-1] + q[1:])
        area12 = 1 / 2 * (s[:-1] + s[1:])
        r12, u12, p12, E12, _ = self.rupEc_from_q(q12, area12)

        h12 = E12 + p12 / r12 + 0.5 * u12 * u12
        a12 = np.sqrt((gamma - 1.0) * (h12 - 0.5 * u12 * u12))

        # Auxiliary variables used to compute P_{j+1/2}^{-1}
        alph112 = (gamma - 1.0) * u12 * u12 / (2 * a12 * a12)
        alph212 = (gamma - 1.0) / (a12 * a12)

        # Compute vector (W_{j+1}-W_j)
        w12 = np.roll(q, -1, axis=0) - q

        # Initialize Roe flux
        df = np.zeros((nx - 1, 3))

        for j in range(0, nx - 1):

            umoy = u12[j]  # {hat U}_{j+1/2}
            hmoy = h12[j]  # {hat H}_{j+1/2}
            amoy = a12[j]  # {hat a}_{j+1/2}

            alph1 = alph112[j]
            alph2 = alph212[j]

            # Compute matrix P^{-1}_{j+1/2}
            Pinv = np.array(
                [
                    [0.5 * (alph1 + umoy / amoy), -0.5 * (alph2 * umoy + 1 / amoy), alph2 / 2],
                    [1 - alph1, alph2 * umoy, -alph2],
                    [0.5 * (alph1 - umoy / amoy), -0.5 * (alph2 * umoy - 1 / amoy), alph2 / 2],
                ]
            )

            # Compute matrix P_{j+1/2}
            P = np.array(
                [
                    [1, 1, 1],
                    [umoy - amoy, umoy, umoy + amoy],
                    [hmoy - amoy * umoy, 0.5 * umoy * umoy, hmoy + amoy * umoy],
                ]
            )

            # Compute matrix Lambda_{j+1/2}
            lamb = np.array([[abs(umoy - amoy), 0, 0], [0, abs(umoy), 0], [0, 0, abs(umoy + amoy)]])

            # Compute Roe matrix |A_{j+1/2}|
            A = np.dot(P, lamb)
            A = np.dot(A, Pinv)

            # Compute |A_{j+1/2}| (W_{j+1}-W_j)
            wdif = w12[j, :]
            df[j, :] = np.dot(A, wdif)

        # ==============================================================
        # Compute df=(f(W_{j+1}+f(W_j))/2-|A_{j+1/2}| (W_{j+1}-W_j)/2
        # ==============================================================
        f = self.flux_from_q(q, s)
        df = 0.5 * (f[0:-1] + f[1:]) - 0.5 * df

        return df

    def mach_from_q(self, q, s):
        _, u, _, _, c = self.rupEc_from_q(q, s)
        return u / c

    def rupEc_from_q(self, q, s):
        gamma = self.gas.gamma(0.0)
        if q.ndim == 1:
            r = q[0] / s
            u = q[1] / q[0]
            E = q[2] / q[0]
        else:
            r = q[:, 0] / s
            u = q[:, 1] / q[:, 0]
            E = q[:, 2] / q[:, 0]

        p = (gamma - 1.0) * r * (E - 0.5 * u * u)
        c = np.sqrt(gamma * p / r)
        return r, u, p, E, c

    def flux_from_q(self, q, s):
        r, u, p, E, _ = self.rupEc_from_q(q, s)
        w = s * r * u  # mass
        h = E + p / r  # enthalpie
        m = s * r * u**2  # momentum
        f1 = w
        f2 = m + s * p
        f3 = w * h
        return np.transpose(np.array([f1, f2, f3]))

    def wpt_from_q(self, q, area):
        gas = self.gas
        if q.ndim == 2:
            w = q[:, 1]
        else:
            w = q[1]

        r, u, ps, E, c = self.rupEc_from_q(q, area)
        mach = u / c
        h = E + ps / r - 0.5 * u * u
        ts = gas.t_from_h(h)
        tt = gas.total_t(ts, mach)
        pt = gas.total_p(ps, ts, tt)
        return w, pt, tt

    def q_from_wpt(self, w, pt, tt, area, subsonic):
        gas = self.gas
        ru = w / area

        if isinstance(w, type(np.zeros((1)))):
            mach = np.array(
                [gas.mach(pt[i], tt[i], ru[i], subsonic=subsonic) for i in range(0, len(w))]
            )
        else:
            mach = gas.mach(pt, tt, ru, subsonic=subsonic)

        ps = gas.static_p(pt, tt, mach)
        ts = gas.static_t(tt, mach)
        r = gas.density(ps, ts)
        u = ru / r
        rE = r * (gas.h(ts) + 0.5 * u * u) - ps

        q = np.transpose(area * np.array([r, ru, rE]))
        return q

    def q_from_fluid_port(self, fl, area, subsonic):
        return self.q_from_wpt(fl.W, fl.Pt, fl.Tt, area, subsonic)
