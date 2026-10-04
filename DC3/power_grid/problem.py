"""Economic MPC for a PV + BESS microgrid, transcribed from ``problems/power_grid``.

Source of truth
---------------
``problems/power_grid/problem.jl`` (``energy_mag``, ``struct eco_mpc``) and
``problems/power_grid/jump_solver.jl`` (``mpc_eco_solver``).  With ``N = 96``, ``dT = 0.25 h``:

    variables      m_k (grid import, kW), u_k (BESS power, kW),
                   p_k (delivered power, kW)  for k = 1..N,
                   x_k (state of charge)      for k = 0..N
    parameters     x0, load_{1..N}, gen_{1..N}
    dynamics       x_k = A x_{k-1} + B u_k,   A = 1, B = -dT/BESS
    boundary       x_0 = x0,  x_N >= x_end_min (= 0.5)
    power flow     u_k + m_k + gen_k - load_k - p_k = 0
    bounds         u_min <= u_k <= u_max, p_k >= 0, x_min <= x_k <= x_max, m free
    objective      sum_k r_ec*dT*(m_k + (1-eta)/(2 sqrt(eta)) |u_k|)
                        + r_op * max(m_k, 0)
                        + r_df * max(a/p_k - 1, 0)

The JuMP model writes the three non-smooth terms in epigraph form
(``su >= |u|``, ``sm >= max(m,0)``, ``sd >= max(a/p - 1, 0)``), which at the
optimum equals the closed form above; that closed form is also the
``model === nothing`` branch of ``(obj::eco_mpc)(m, u, p, model)`` and is what is
implemented here.  `validate.py` checks the nominal instance against the
hard-coded ``Jopt`` in ``problems/power_grid/setup.jl``.

DC3 structure
-------------
``x_0`` is a known parameter, so the decision vector is

    y = [ m_1..m_N | u_1..u_N | p_1..p_N | x_1..x_N ]   (n_y = 4N)

and the 2N equalities are exactly the rows of the matrix ``M`` built by
``utils.jl::dynamics_projection``.  The terminal bound ``x_N >= x_end_min`` is an
inequality (row 5N of ``G``).  The partial/dependent split is

    partial   P = { u_1..u_N } U { p_1..p_N }          (2N variables)
    dependent D = { x_1..x_N } U { m_1..m_N }          (2N variables)

which is invertible and in fact block-triangular:

    x_k = x_0 + B * sum_{i<=k} u_i         (k = 1..N, forward recursion)
    m_k = load_k - gen_k - u_k + p_k       (power-flow rows)

The code does not exploit that structure: it uses the generic linear completion
``y_D = A_D^{-1}(b_eq - A_P y_P)`` and *checks* the conditioning of ``A_D``;
`validate.py` asserts that the generic solve reproduces the closed form above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..common.problem import ParametricProblem


# --- system constants, copied verbatim from `energy_mag()` ------------------
DEFAULTS = dict(
    dT=0.25, A=1.0, BESS=500.0, r_ec=0.1, r_df=10.0, r_op=19.19, eta=0.8, a=50.0,
    x_min=0.2, x_max=0.8, x_end_min=0.5, u_min=-700.0, u_max=700.0, x0=0.5, N=96,
)


@dataclass
class GridParams:
    """A batch of instances.  ``x0``: (B,); ``load``/``gen``: (B, N)."""

    x0: torch.Tensor
    load: torch.Tensor
    gen: torch.Tensor
    offsets: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return self.x0.shape[0]

    def index(self, idx) -> "GridParams":
        return GridParams(x0=self.x0[idx], load=self.load[idx], gen=self.gen[idx],
                          offsets=None if self.offsets is None else self.offsets[idx])

    def to(self, device, dtype) -> "GridParams":
        return GridParams(
            x0=self.x0.to(device=device, dtype=dtype),
            load=self.load.to(device=device, dtype=dtype),
            gen=self.gen.to(device=device, dtype=dtype),
            offsets=None if self.offsets is None else self.offsets.to(device=device),
        )


def index_grid(params: GridParams, idx) -> GridParams:
    return params.index(idx)


class EcoMPCProblem(ParametricProblem):
    name = "eco_mpc"

    def __init__(
        self,
        N: int = 96,
        feature_mode: str = "x0_load_gen",
        p_margin: float = 1e-3,
        bound_margin: float = 0.0,
        p_safe_eps: float = 1e-9,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
        **overrides,
    ):
        cst = dict(DEFAULTS)
        cst.update(overrides)
        cst["N"] = int(N)
        self.c = cst
        self.N = int(N)
        self.Ad = float(cst["A"])
        self.Bd = -float(cst["dT"]) / float(cst["BESS"])
        self.kappa = (1.0 - cst["eta"]) / (2.0 * np.sqrt(cst["eta"]))
        self.p_margin = float(p_margin)
        self.bound_margin = float(bound_margin)
        self.p_safe_eps = float(p_safe_eps)
        self.feature_mode = feature_mode
        self.dtype, self.device = dtype, device

        N_ = self.N
        self.n_y = 4 * N_
        self.n_eq = 2 * N_
        self.n_ineq = 5 * N_ + 1
        self.x_dim = {"x0_load_gen": 2 * N_ + 1, "x0_netload": N_ + 1}[feature_mode]

        # slices into y = [m | u | p | x]
        self.sl_m = slice(0, N_)
        self.sl_u = slice(N_, 2 * N_)
        self.sl_p = slice(2 * N_, 3 * N_)
        self.sl_x = slice(3 * N_, 4 * N_)

        self.A_eq = self._build_A_eq().to(device=device, dtype=dtype)
        self.G, self.g_const = self._build_G()
        self.G = self.G.to(device=device, dtype=dtype)
        self.g_const = self.g_const.to(device=device, dtype=dtype)
        self._G_eff_cache: dict[int, torch.Tensor] = {}

    # -- structural matrices ----------------------------------------------
    def _build_A_eq(self) -> torch.Tensor:
        N_ = self.N
        A = torch.zeros(2 * N_, 4 * N_, dtype=torch.float64)
        # dynamics rows 0..N-1 :  Ad*x_{k-1} + Bd*u_k - x_k = rhs_k
        for k in range(N_):
            A[k, self.sl_u.start + k] = self.Bd
            A[k, self.sl_x.start + k] = -1.0
            if k >= 1:
                A[k, self.sl_x.start + k - 1] = self.Ad
        # power-flow rows N..2N-1 :  m_k + u_k - p_k = load_k - gen_k
        for k in range(N_):
            A[N_ + k, self.sl_m.start + k] = 1.0
            A[N_ + k, self.sl_u.start + k] = 1.0
            A[N_ + k, self.sl_p.start + k] = -1.0
        return A

    def _build_G(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Inequalities as ``G y + g_const <= 0`` (5N+1 rows, instance independent)."""
        N_, c = self.N, self.c
        G = torch.zeros(5 * N_ + 1, 4 * N_, dtype=torch.float64)
        g = torch.zeros(5 * N_ + 1, dtype=torch.float64)
        eye = torch.eye(N_, dtype=torch.float64)
        # u <= u_max
        G[0:N_, self.sl_u] = eye;              g[0:N_] = -c["u_max"]
        # u_min <= u
        G[N_:2 * N_, self.sl_u] = -eye;        g[N_:2 * N_] = c["u_min"]
        # p >= 0
        G[2 * N_:3 * N_, self.sl_p] = -eye;    g[2 * N_:3 * N_] = 0.0
        # x <= x_max
        G[3 * N_:4 * N_, self.sl_x] = eye;     g[3 * N_:4 * N_] = -c["x_max"]
        # x_min <= x
        G[4 * N_:5 * N_, self.sl_x] = -eye;    g[4 * N_:5 * N_] = c["x_min"]
        # x_end_min <= x_N  (terminal bound)
        G[5 * N_, self.sl_x.stop - 1] = -1.0;  g[5 * N_] = c["x_end_min"]
        return G, g

    def default_other_vars(self) -> list[int]:
        """The structured dependent set { x } U { m } (see class docstring)."""
        return (list(range(self.sl_x.start, self.sl_x.stop))
                + list(range(self.sl_m.start, self.sl_m.stop)))

    # -- instance plumbing -------------------------------------------------
    def features(self, p: GridParams) -> torch.Tensor:
        if self.feature_mode == "x0_netload":
            return torch.cat([p.x0.unsqueeze(1), p.load - p.gen], dim=1)
        return torch.cat([p.x0.unsqueeze(1), p.load, p.gen], dim=1)

    def eq_rhs(self, p: GridParams) -> torch.Tensor:
        B = len(p)
        N_ = self.N
        rhs = torch.zeros(B, 2 * N_, dtype=p.x0.dtype, device=p.x0.device)
        rhs[:, 0] = -self.Ad * p.x0                 # first dynamics row
        rhs[:, N_:] = p.load - p.gen                # power-flow rows
        return rhs

    # -- objective ---------------------------------------------------------
    def obj_fn(self, p: GridParams, Y: torch.Tensor, safe: bool = True) -> torch.Tensor:
        c = self.c
        m, u, pw = Y[:, self.sl_m], Y[:, self.sl_u], Y[:, self.sl_p]
        pw_eval = torch.clamp(pw, min=self.p_safe_eps) if safe else pw
        energy = c["r_ec"] * c["dT"] * (m + self.kappa * u.abs())
        peak = c["r_op"] * torch.clamp(m, min=0.0)
        discomfort = c["r_df"] * torch.clamp(c["a"] / pw_eval - 1.0, min=0.0)
        return (energy + peak + discomfort).sum(dim=1)

    # -- constraints -------------------------------------------------------
    def ineq_resid(self, p: GridParams, Y: torch.Tensor, margin: bool = False) -> torch.Tensor:
        r = Y @ self.G.T + self.g_const
        if margin:
            N_ = self.N
            add = torch.zeros_like(self.g_const)
            add[0:2 * N_] = self.bound_margin
            add[2 * N_:3 * N_] = self.p_margin       # p >= p_margin
            add[3 * N_:] = self.bound_margin
            r = r + add
        return r

    def domain_resid(self, p: GridParams, Y: torch.Tensor) -> torch.Tensor:
        """``a/p`` needs ``p > 0``; returns ``-p`` (positive = outside the domain)."""
        return -Y[:, self.sl_p]

    def domain_valid(self, p: GridParams, Y: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(Y).all(dim=1) & (Y[:, self.sl_p] > 0).all(dim=1)

    # -- closed-form correction gradient ----------------------------------
    def _G_eff(self, completion) -> torch.Tensor:
        key = id(completion)
        if key not in self._G_eff_cache:
            G_P = self.G[:, completion.partial_vars]
            G_D = self.G[:, completion.other_vars]
            self._G_eff_cache[key] = G_P - G_D @ completion.A_other_inv_A_partial
        return self._G_eff_cache[key]

    def ineq_row_scale(self, completion, zero_tol: float = 1e-9):
        """``1/||G_eff_i||``, normalised to a median of 1.

        ``G_eff = G_P - G_D A_D^{-1} A_P`` is constant here, so this is exact.
        The state-of-charge rows come out ~2000x larger than the power rows,
        which is exactly the factor ``1/|B| = BESS/dT`` that makes the unscaled
        correction useless.

        A row whose reduced gradient is identically **zero** depends only on the
        instance parameters, so no choice of partial variables can change it.
        Such rows (none in the default partition) are given weight 1 instead of
        ``1/0``; amplifying their round-off would otherwise manufacture spurious
        "violations" of size ``corr_eps``.
        """
        norms = self._G_eff(completion).norm(dim=1)
        nz = norms > zero_tol * norms.max()
        scale = torch.ones_like(norms)
        scale[nz] = 1.0 / norms[nz]
        scale = scale / scale[nz].median()
        scale[~nz] = 1.0
        return scale.to(dtype=torch.float64)

    def ineq_partial_grad(self, p: GridParams, Z: torch.Tensor, completion,
                          row_scale=None) -> torch.Tensor:
        Y = completion.complete(Z, self.eq_rhs(p))
        r = self.ineq_dist(p, Y, margin=True)
        if row_scale is not None:
            r = r * (row_scale ** 2)      # d/dZ ||s*r||^2 = 2 (s^2 r)^T G_eff
        return 2.0 * (r @ self._G_eff(completion))

    # -- bounded output parametrisation ------------------------------------
    def partial_bounds(self, completion):
        """`u` gets its real box [u_min, u_max]; `p` gets only a lower bound.

        Every other partial variable (there are none in the default partition)
        is left unconstrained.  `p` deliberately gets *no* upper bound: the
        power-flow equality makes it unbounded above, and clipping it could cut
        off the optimum.
        """
        lo = torch.full((completion.n_partial,), -float("inf"), dtype=torch.float64)
        hi = torch.full((completion.n_partial,), float("inf"), dtype=torch.float64)
        idx = completion.partial_vars.cpu().numpy()
        for j, v in enumerate(idx):
            if self.sl_u.start <= v < self.sl_u.stop:
                lo[j], hi[j] = self.c["u_min"], self.c["u_max"]
            elif self.sl_p.start <= v < self.sl_p.stop:
                lo[j] = max(self.p_margin, 1e-9)
        return lo, hi

    def partial_init_target(self, completion):
        """Start at `u = 0` (idle battery) and `p = a` (zero discomfort)."""
        t = torch.zeros(completion.n_partial, dtype=torch.float64)
        idx = completion.partial_vars.cpu().numpy()
        for j, v in enumerate(idx):
            if self.sl_p.start <= v < self.sl_p.stop:
                t[j] = self.c["a"]
        return t

    # -- reporting ---------------------------------------------------------
    def unpack(self, Y: torch.Tensor) -> dict:
        return {"m": Y[:, self.sl_m], "u": Y[:, self.sl_u],
                "p": Y[:, self.sl_p], "x": Y[:, self.sl_x]}

    def to(self, device, dtype):
        self.A_eq = self.A_eq.to(device=device, dtype=dtype)
        self.G = self.G.to(device=device, dtype=dtype)
        self.g_const = self.g_const.to(device=device, dtype=dtype)
        self._G_eff_cache.clear()
        self.device, self.dtype = device, dtype
        return self
