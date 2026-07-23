"""Optimized trapezoidal solver for the driven Fermi-polaron equation.

The program solves Eq. (6) of the accompanying manuscript for chi_q(t),
and evaluates the contact C(t), energy E(t), and residue Z(t) through
Eqs. (9)-(11).

Numerical structure retained from the original program
------------------------------------------------------
1. The short-time asymptotic series supplies the first ``n_seed`` points.
2. The singular convolution in L-hat uses the same Hadamard endpoint
   corrections at tau -> 0 and tau -> t.
3. The phi_0 integral uses the trapezoidal rule in both time and momentum.

Main optimizations
------------------
1. M is diagonal plus rank one, so the Sherman-Morrison formula replaces a
   dense linear solve at every time step.
2. All result arrays are preallocated; no np.append/np.hstack is used.
3. q-dependent constants and the regular convolution kernel are precomputed.
4. The tau -> 0 correction evaluates cf_0, cf_1, and cf_2 only once per step.
5. E(t) is accumulated with a second-order trapezoidal rule and np.cumsum.

The remaining history convolution is evaluated directly and exactly for the
chosen discretization. Its cost is O(n_q*n_t**2); for substantially larger
n_t, a block-online FFT convolution is the natural next optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import special as sps
from tqdm import tqdm


@dataclass(frozen=True)
class SolverConfig:
    """Physical and numerical parameters."""

    beta: float = 2.0 * np.sqrt(2.0 / np.pi) * 3.0 / 10.0
    kf: float = 1.0
    n_q: int = 100
    dt: float = 1.0 / 200.0
    n_t: int = 24_000
    n_seed: int = 14
    show_progress: bool = True

    @property
    def dq(self) -> float:
        return self.kf / self.n_q


@dataclass
class ShortTimeCoefficients:
    """q-dependent coefficients used by the endpoint corrections."""

    cx1: np.ndarray
    cx3: np.ndarray
    cx5: np.ndarray
    cx6: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    a3: np.ndarray
    a4: np.ndarray


@dataclass
class SimulationResult:
    time: np.ndarray
    q: np.ndarray
    chi_qt: np.ndarray
    phi0: np.ndarray
    contact: np.ndarray
    energy: np.ndarray
    residue: np.ndarray
    min_rank1_denominator: float
    elapsed_seconds: float


# Riemann-zeta values occur repeatedly in the Hadamard corrections.
ZETA_M_HALF = sps.zeta(-0.5)
ZETA_M_3HALF = sps.zeta(-1.5)
ZETA_M_5HALF = sps.zeta(-2.5)
ZETA_M_3 = sps.zeta(-3.0)
ZETA_HALF = sps.zeta(0.5)
ZETA_3HALF = sps.zeta(1.5)


def scattering_length(t: float | np.ndarray, beta: float) -> np.ndarray:
    """Ramp a_s(t) = beta*sqrt(t)."""

    return beta * np.sqrt(t)


def trapezoidal_q_weights(config: SolverConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return q grid and weights for integral_0^kF dq q^2 g(q).

    With q_p = p*dq,

        integral dq q^2 g(q) ~= dq^3 sum_p w_p p^2 g(q_p).

    The q=0 endpoint has zero weight because of p^2, while q=kF carries
    the usual trapezoidal factor 1/2.
    """

    p = np.arange(config.n_q + 1, dtype=float)
    q = p * config.dq
    weights = p**2
    weights[-1] *= 0.5
    return q, weights


def build_short_time_coefficients(
    config: SolverConfig, q: np.ndarray
) -> ShortTimeCoefficients:
    """Precompute all q-dependent asymptotic and endpoint coefficients."""

    beta = config.beta
    kf = config.kf
    dt = config.dt
    sqrt_dt = np.sqrt(dt)
    common = (4.0j * np.pi) ** (-1.5)

    # Coefficients in the small-s expansion of f_q(s), s=t-tau.
    af_n3 = np.full(q.shape, common, dtype=np.complex128)
    af_n1 = 1.0j * common * (q / 2.0) ** 2
    af_0 = np.full(q.shape, -(kf**3) / (6.0 * np.pi**2), dtype=np.complex128)
    af_1 = -0.5 * common * (q / 2.0) ** 4
    af_2 = np.full(q.shape, 1.0j * kf**5 / (10.0 * np.pi**2), dtype=np.complex128)
    af_3 = -(1.0j / 6.0) * common * (q / 2.0) ** 6

    # Hadamard corrections at tau -> t. These are algebraically identical
    # to a1(q), ..., a4(q) in the original script.
    a1 = (
        dt * af_0 / 4.0
        - af_n3 * ZETA_M_3HALF / (2.0 * sqrt_dt)
        + 2.5 * af_n1 * sqrt_dt * ZETA_M_3HALF
        - 3.0 * af_1 * dt**1.5 * ZETA_M_3HALF
        + 2.5 * af_n3 * ZETA_M_HALF / sqrt_dt
        - 3.0 * af_n1 * sqrt_dt * ZETA_M_HALF
        - 3.0 * af_n3 * ZETA_HALF / sqrt_dt
    )

    a2 = (
        -dt * af_0 / 8.0
        + af_n3 * ZETA_M_3HALF / (2.0 * sqrt_dt)
        - 2.0 * af_n1 * sqrt_dt * ZETA_M_3HALF
        + 1.5 * af_1 * dt**1.5 * ZETA_M_3HALF
        - 2.0 * af_n3 * ZETA_M_HALF / sqrt_dt
        + 1.5 * af_n1 * sqrt_dt * ZETA_M_HALF
        + 1.5 * af_n3 * ZETA_HALF / sqrt_dt
    )

    a3 = (
        dt * af_0 / 36.0
        - af_n3 * ZETA_M_3HALF / (6.0 * sqrt_dt)
        + 0.5 * af_n1 * sqrt_dt * ZETA_M_3HALF
        - af_1 * dt**1.5 * ZETA_M_3HALF / 3.0
        + 0.5 * af_n3 * ZETA_M_HALF / sqrt_dt
        - af_n1 * sqrt_dt * ZETA_M_HALF / 3.0
        - af_n3 * ZETA_HALF / (3.0 * sqrt_dt)
    )

    a4 = (
        25.0 * dt * af_0 / 72.0
        + dt**2 * af_2 / 12.0
        + af_n3 * ZETA_M_3HALF / (6.0 * sqrt_dt)
        - af_n1 * sqrt_dt * ZETA_M_3HALF
        + (11.0 / 6.0) * af_1 * dt**1.5 * ZETA_M_3HALF
        - af_3 * dt**2.5 * ZETA_M_3HALF
        - af_n3 * ZETA_M_HALF / sqrt_dt
        + (11.0 / 6.0) * af_n1 * sqrt_dt * ZETA_M_HALF
        - af_1 * dt**1.5 * ZETA_M_HALF
        + (11.0 / 6.0) * af_n3 * ZETA_HALF / sqrt_dt
        - af_n1 * sqrt_dt * ZETA_HALF
        - af_n3 * ZETA_3HALF / sqrt_dt
    )

    # Coefficients in chi_q(t) at t -> 0.
    cx1_scalar = 4.0 * np.pi / (
        1.0 / beta + sps.beta(1.5, -0.5) / np.sqrt(4.0j * np.pi)
    )
    cx1 = np.full(q.shape, cx1_scalar, dtype=np.complex128)

    cx3 = (
        -1.0j
        / np.sqrt(4.0j * np.pi)
        * sps.beta(1.5, 0.5)
        / (1.0 / beta + sps.beta(2.5, -0.5) / np.sqrt(4.0j * np.pi))
        * (q / 2.0) ** 2
        * cx1
    )

    cx5 = (
        -1.0
        / np.sqrt(4.0j * np.pi)
        * (
            1.0j * (q / 2.0) ** 2 * sps.beta(2.5, 0.5) * cx3
            - 0.5 * (q / 2.0) ** 4 * sps.beta(1.5, 1.5) * cx1
        )
        / (1.0 / beta + sps.beta(3.5, -0.5) / np.sqrt(4.0j * np.pi))
    )

    cx6 = (
        4.0j
        * np.pi
        / (1.0 / beta + sps.beta(4.0, -0.5) / np.sqrt(4.0j * np.pi))
        * (
            -kf**5
            / (25.0 * np.pi**2)
            * 0.25
            * (
                -1.0j
                / np.sqrt(4.0j * np.pi)
                * sps.beta(1.5, 0.5)
                / (
                    1.0 / beta
                    + sps.beta(2.5, -0.5) / np.sqrt(4.0j * np.pi)
                )
                * cx1
            )
            + 2.0 * kf**3 * cx3 / (30.0 * np.pi**2)
            - 1.0j
            * kf**5
            * cx1
            * sps.beta(1.5, 2.0)
            / (10.0 * np.pi**2)
        )
    )

    return ShortTimeCoefficients(cx1, cx3, cx5, cx6, a1, a2, a3, a4)


def short_time_chi(
    t: np.ndarray, coeff: ShortTimeCoefficients
) -> np.ndarray:
    """Return chi_q(t) from the short-time series.

    Output shape is (n_q+1, len(t)).
    """

    t = np.asarray(t, dtype=float)[None, :]
    return (
        coeff.cx1[:, None] * t**0.5
        + coeff.cx3[:, None] * t**1.5
        + coeff.cx5[:, None] * t**2.5
        + coeff.cx6[:, None] * t**3
    )


def kernel_f_all_q(t: float, q: np.ndarray, kf: float) -> np.ndarray:
    """Evaluate f_q(t) for the complete q grid in one vectorized call."""

    result = np.empty(q.shape, dtype=np.complex128)
    sqrt_it = np.sqrt(1.0j * t)

    # q=0 has a finite limiting expression and is treated separately.
    result[0] = (
        -2.0j * np.exp(-1.0j * kf**2 * t) * kf / t
        + np.sqrt(np.pi) * sps.erfc(kf * sqrt_it) / (1.0j * t) ** 1.5
    ) / (8.0 * np.pi**2)

    qp = q[1:]
    # expm1 avoids cancellation in 2-2*exp(2*i*kF*q*t) at small q*t.
    first_difference = -2.0 * np.expm1(2.0j * kf * qp * t)
    erfc_sum = sps.erfc((2.0 * kf - qp) * sqrt_it / 2.0) + sps.erfc(
        (2.0 * kf + qp) * sqrt_it / 2.0
    )
    result[1:] = (
        np.exp(-1.0j * kf * (kf + qp) * t)
        * (
            first_difference
            - np.sqrt(1.0j * np.pi * t)
            * qp
            * np.exp(1.0j * (kf + qp / 2.0) ** 2 * t)
            * erfc_sum
        )
        / (16.0 * np.pi**2 * qp * t**2)
    )
    return result


def cf_all_q(
    t: float, q: np.ndarray, kf: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized cf_0, cf_1, and cf_2 used at the tau -> 0 endpoint."""

    cf0 = np.empty(q.shape, dtype=np.complex128)
    cf1 = np.empty(q.shape, dtype=np.complex128)
    cf2 = np.empty(q.shape, dtype=np.complex128)

    sqrt_it = np.sqrt(1.0j * t)
    exp_k = np.exp(-1.0j * kf**2 * t)
    erfc_0 = sps.erfc(kf * sqrt_it)

    cf0[0] = -(
        2.0j * exp_k * kf / t
        + np.sqrt(np.pi) * sqrt_it * erfc_0 / t**2
    ) / (8.0 * np.pi**2)

    cf1[0] = -exp_k * (
        6.0j * kf * t
        - 4.0 * kf**3 * t**2
        + 3.0 * np.exp(1.0j * kf**2 * t) * np.sqrt(1.0j * np.pi * t) * erfc_0
    ) / (16.0 * np.pi**2 * t**3)

    cf2[0] = -exp_k * (
        30.0j * kf * t
        - 20.0 * kf**3 * t**2
        - 8.0j * kf**5 * t**3
        + 15.0 * np.exp(1.0j * kf**2 * t) * np.sqrt(1.0j * np.pi * t) * erfc_0
    ) / (64.0 * np.pi**2 * t**4)

    qp = q[1:]
    q2 = qp**2
    sqrt_i_pi_t = np.sqrt(1.0j * np.pi * t)
    erfc_minus = sps.erfc((2.0 * kf - qp) * sqrt_it / 2.0)
    erfc_plus = sps.erfc((2.0 * kf + qp) * sqrt_it / 2.0)

    cf0[1:] = (
        np.exp(-1.0j * kf * (kf + qp) * t)
        * (
            -2.0 * np.expm1(2.0j * kf * qp * t)
            - np.exp(1.0j * (kf + qp / 2.0) ** 2 * t)
            * np.sqrt(np.pi)
            * qp
            * sqrt_it
            * (erfc_minus + erfc_plus)
        )
        / (16.0 * np.pi**2 * qp * t**2)
    )

    cf1[1:] = (
        np.exp(-1.0j * (8.0 * kf**2 + q2) * t / 4.0)
        * (
            1.0j
            * np.exp(1.0j * (4.0 * kf**2 + q2) * t / 2.0)
            * sqrt_i_pi_t
            * qp
            * (6.0j + q2 * t)
            * erfc_minus
            + np.exp(1.0j * (-2.0 * kf + qp) ** 2 * t / 4.0)
            * (
                16.0
                + 8.0j * kf**2 * t
                + 4.0j * kf * qp * t
                - 2.0j * q2 * t
                - 2.0j
                * np.exp(2.0j * kf * qp * t)
                * (-8.0j + 4.0 * kf**2 * t - 2.0 * kf * qp * t - q2 * t)
                + 1.0j
                * np.exp(1.0j * (2.0 * kf + qp) ** 2 * t / 4.0)
                * qp
                * sqrt_i_pi_t
                * (6.0j + q2 * t)
                * erfc_plus
            )
        )
        / (64.0 * np.pi**2 * qp * t**3)
    )

    common_poly = -96.0 + 14.0j * q2 * t + q2**2 * t**2
    poly_minus = (
        common_poly
        + 16.0 * kf**4 * t**2
        + 24.0 * kf**3 * qp * t**2
        + 4.0 * kf**2 * t * (-16.0j + q2 * t)
        - 2.0 * kf * qp * t * (18.0j + q2 * t)
    )
    poly_plus = (
        common_poly
        + 16.0 * kf**4 * t**2
        - 24.0 * kf**3 * qp * t**2
        + 4.0 * kf**2 * t * (-16.0j + q2 * t)
        + 2.0 * kf * qp * t * (18.0j + q2 * t)
    )
    erfc_poly = -60.0 + 12.0j * q2 * t + q2**2 * t**2

    cf2[1:] = (
        np.exp(-1.0j * (8.0 * kf**2 + q2) * t / 4.0)
        * (
            -2.0
            * np.exp(1.0j * (-2.0 * kf + qp) ** 2 * t / 4.0)
            * poly_minus
            + 2.0
            * np.exp(1.0j * (2.0 * kf + qp) ** 2 * t / 4.0)
            * poly_plus
            + np.exp(1.0j * (4.0 * kf**2 + q2) * t / 2.0)
            * qp
            * sqrt_i_pi_t
            * erfc_poly
            * (erfc_minus + erfc_plus)
        )
        / (512.0 * np.pi**2 * qp * t**4)
    )

    return cf0, cf1, cf2


def initial_endpoint_correction(
    t: float,
    q: np.ndarray,
    config: SolverConfig,
    coeff: ShortTimeCoefficients,
) -> np.ndarray:
    """Hadamard correction generated by the tau -> 0 endpoint."""

    cf0, cf1, cf2 = cf_all_q(t, q, config.kf)
    c1 = coeff.cx1 * cf0
    c3 = coeff.cx3 * cf0 + coeff.cx1 * cf1
    c5 = coeff.cx5 * cf0 + coeff.cx3 * cf1 + coeff.cx1 * cf2
    c6 = coeff.cx6 * cf0

    dt = config.dt
    return -(
        c1 * ZETA_M_HALF * dt**1.5
        + c3 * ZETA_M_3HALF * dt**2.5
        + c5 * ZETA_M_5HALF * dt**3.5
        + c6 * ZETA_M_3 * dt**4
    )


def build_effective_history_kernel(
    config: SolverConfig,
    q: np.ndarray,
    coeff: ShortTimeCoefficients,
) -> np.ndarray:
    """Precompute dt*f_q(lag*dt) and add the tau -> t corrections.

    Column ``lag-1`` corresponds to a positive time lag ``lag*dt``. The
    a1/a2/a3 corrections therefore belong to columns 0/1/2.
    """

    kernel = np.empty((q.size, config.n_t - 1), dtype=np.complex128)
    iterator = range(1, config.n_t)
    if config.show_progress:
        iterator = tqdm(iterator, desc="Precomputing kernel", leave=False)

    for lag in iterator:
        kernel[:, lag - 1] = config.dt * kernel_f_all_q(
            lag * config.dt, q, config.kf
        )

    kernel[:, 0] += coeff.a1
    kernel[:, 1] += coeff.a2
    kernel[:, 2] += coeff.a3
    return kernel


def solve_rank_one_system(
    diagonal: np.ndarray,
    rank_one_row: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, complex]:
    """Solve (diag(diagonal) + 1*v^T)x = rhs exactly."""

    inv_diagonal = 1.0 / diagonal
    y = inv_diagonal * rhs
    z = inv_diagonal  # D^{-1} times the all-ones column
    denominator = 1.0 + np.dot(rank_one_row, z)
    solution = y - z * np.dot(rank_one_row, y) / denominator
    return solution, denominator


def short_time_phi0(t: np.ndarray, beta: float) -> np.ndarray:
    """Short-time phi_0 used in the original program."""

    denominator = np.exp(3.0j * np.pi / 4.0) * np.sqrt(np.pi) * beta + 2.0
    term_3half = (
        -1.0j * 8.0 * beta / (9.0 * np.pi * denominator) * t**1.5
    )
    term_5half = (
        (-3.0j * np.sqrt(np.pi) * beta + 4.0 * np.exp(3.0j * np.pi / 4.0))
        / (200.0 * np.sqrt(np.pi) * denominator)
        * t**2.5
    )
    return 1.0 + term_3half + term_5half


def short_time_energy_reference(t: np.ndarray) -> np.ndarray:
    """Original short-time E(t) for beta=0.3*beta_c1 and kF=1.

    This closed expression is retained to reproduce the original calculation.
    If beta or kF is changed, replace it with the corresponding general
    short-time expression before using it as the energy matching condition.
    """

    return (
        2.0
        * np.sqrt(t)
        * (4_595_500.0 - 136_500.0 * t + 5_247.0 * t**2)
        / (6_663_475.0 * np.sqrt(2.0) * np.pi**1.5)
    )


def short_time_contact_reference(t: np.ndarray) -> np.ndarray:
    """Original short-time C(t) for beta=0.3*beta_c1 and kF=1."""

    return (
        96.0 / (29.0 * np.pi) * t
        - 864.0 / (2_929.0 * np.pi) * t**2
        + 125_928.0 / (6_663_475.0 * np.pi) * t**3
    )


def run_simulation(config: SolverConfig) -> SimulationResult:
    """Solve for chi_q(t), C(t), E(t), and Z(t)."""

    if config.n_seed < 3:
        raise ValueError("n_seed must be at least 3 for the a1/a2/a3 corrections.")
    if config.n_seed >= config.n_t:
        raise ValueError("n_seed must be smaller than n_t.")

    q, q_weights = trapezoidal_q_weights(config)
    time_grid = config.dt * np.arange(1, config.n_t + 1, dtype=float)
    coeff = build_short_time_coefficients(config, q)
    kernel = build_effective_history_kernel(config, q, coeff)

    # q_measure implements (1/2pi^2) integral_0^kF dq q^2.
    q_measure = config.dq**3 * q_weights / (2.0 * np.pi**2)

    # wq_full is i*dt times the q integral. The current time endpoint carries
    # half of this weight in the trapezoidal rule.
    wq_full = 1.0j * config.dt * q_measure
    rank_one_row = 0.5 * wq_full

    chi_qt = np.empty((q.size, config.n_t), dtype=np.complex128)
    phi0 = np.empty(config.n_t, dtype=np.complex128)

    seed_time = time_grid[: config.n_seed]
    chi_qt[:, : config.n_seed] = short_time_chi(seed_time, coeff)
    phi0[: config.n_seed] = short_time_phi0(seed_time, config.beta)

    # This scalar stores full-weight contributions from all already-known
    # positive-time points. The half-weight current point is not included.
    i_phi0_history = np.dot(
        np.sum(chi_qt[:, : config.n_seed], axis=1), wq_full
    )

    min_denominator = np.inf
    start = time.perf_counter()
    iterator = range(config.n_seed, config.n_t)
    if config.show_progress:
        iterator = tqdm(iterator, desc="Solving chi_q(t)")

    for n in iterator:
        t_n = time_grid[n]

        # Direct causal convolution over all earlier times. The first three
        # lag columns already contain the a1/a2/a3 endpoint corrections.
        convolution = np.einsum(
            "qm,qm->q",
            chi_qt[:, :n],
            kernel[:, n - 1 :: -1],
            optimize=False,
        )
        b0 = initial_endpoint_correction(t_n, q, config, coeff)

        rhs = 1.0 - i_phi0_history - 1.0j * (convolution + b0)
        diagonal = (
            1.0 / (4.0 * np.pi * scattering_length(t_n, config.beta))
            + 1.0j * coeff.a4
        )

        chi_current, denominator = solve_rank_one_system(
            diagonal, rank_one_row, rhs
        )
        min_denominator = min(min_denominator, abs(denominator))
        chi_qt[:, n] = chi_current

        # Current phi_0 contains half the current endpoint; afterwards the
        # current point becomes a full-weight history point for step n+1.
        current_increment = np.dot(chi_current, wq_full)
        phi0[n] = 1.0 - i_phi0_history - 0.5 * current_increment
        i_phi0_history += current_increment

    elapsed = time.perf_counter() - start

    contact = np.sum(np.abs(chi_qt) ** 2 * q_measure[:, None], axis=0)
    residue = np.abs(phi0) ** 2

    # Tan sweep theorem for a_s(t)=beta*sqrt(t):
    # dE/dt = C(t)/(8*pi*beta*t^(3/2)).
    energy_rate = contact / (
        8.0 * np.pi * config.beta * time_grid**1.5
    )
    energy = np.empty(config.n_t, dtype=float)
    energy[: config.n_seed] = short_time_energy_reference(seed_time)
    increments = 0.5 * config.dt * (
        energy_rate[config.n_seed - 1 : -1]
        + energy_rate[config.n_seed :]
    )
    energy[config.n_seed :] = (
        energy[config.n_seed - 1] + np.cumsum(increments)
    )

    return SimulationResult(
        time=time_grid,
        q=q,
        chi_qt=chi_qt,
        phi0=phi0,
        contact=contact,
        energy=energy,
        residue=residue,
        min_rank1_denominator=float(min_denominator),
        elapsed_seconds=elapsed,
    )


def validate_result(result: SimulationResult) -> None:
    """Basic diagnostics that catch indexing and floating-point failures."""

    arrays = {
        "chi_qt": result.chi_qt,
        "phi0": result.phi0,
        "contact": result.contact,
        "energy": result.energy,
        "residue": result.residue,
    }
    for name, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"{name} contains NaN or infinity.")

    if np.min(result.contact) < -1e-12:
        raise FloatingPointError("Contact became negative beyond roundoff.")


def save_result(
    result: SimulationResult, config: SolverConfig, output: Path
) -> None:
    """Save arrays and numerical parameters in one compressed NPZ file."""

    np.savez_compressed(
        output,
        time=result.time,
        q=result.q,
        chi_qt=result.chi_qt,
        phi0=result.phi0,
        contact=result.contact,
        energy=result.energy,
        residue=result.residue,
        beta=config.beta,
        kf=config.kf,
        dq=config.dq,
        dt=config.dt,
        n_q=config.n_q,
        n_t=config.n_t,
        n_seed=config.n_seed,
    )


def plot_result(result: SimulationResult, config: SolverConfig) -> None:
    """Plot C(t), E(t), and Z(t)."""

    beta_c1 = 2.0 * np.sqrt(2.0 / np.pi)
    beta_ratio = config.beta / beta_c1
    title_suffix = (
        rf"$\beta/\beta_{{c1}}={beta_ratio:.3g},\ "
        rf"\Delta t={config.dt:.4g},\ N_q={config.n_q}$"
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 10.0), constrained_layout=True)

    axes[0].plot(result.time, result.contact, color="tab:green", lw=1.0, label="numerical")
    axes[0].plot(
        result.time,
        short_time_contact_reference(result.time),
        color="tab:red",
        lw=1.0,
        label="short-time series",
    )
    axes[0].set_ylabel(r"$C(t)$")
    axes[0].set_ylim(-0.1, 5.5)
    axes[0].legend()
    axes[0].set_title(title_suffix)

    axes[1].plot(result.time, result.energy, color="tab:green", lw=1.0, label="numerical")
    axes[1].plot(
        result.time,
        short_time_energy_reference(result.time),
        color="tab:red",
        lw=1.0,
        label="short-time series",
    )
    axes[1].set_ylabel(r"$E(t)$")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    axes[2].plot(result.time, result.residue, color="tab:green", lw=1.0, label="numerical")
    axes[2].plot(
        result.time,
        np.abs(short_time_phi0(result.time, config.beta)) ** 2,
        color="tab:red",
        lw=1.0,
        label="short-time series",
    )
    axes[2].set_xlabel(r"$t$")
    axes[2].set_ylabel(r"$Z(t)$")
    axes[2].set_ylim(0.0, 1.1)
    axes[2].legend()

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use a small grid to check that the program runs correctly.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Do not open Matplotlib figures."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fermi_polaron_trapezoid_results.npz"),
        help="Output NPZ filename.",
    )
    # Jupyter/IPython automatically appends an argument of the form
    # ``--f=.../kernel-xxxx.json``.  parse_known_args keeps the normal
    # command-line options above while safely ignoring that kernel argument.
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    if args.quick_test:
        config = SolverConfig(n_q=20, n_t=120, n_seed=14, show_progress=False)
    else:
        config = SolverConfig()

    result = run_simulation(config)
    validate_result(result)
    save_result(result, config, args.output)

    print(f"Evolution time: {result.elapsed_seconds:.3f} s")
    print(
        "Minimum |Sherman-Morrison denominator|: "
        f"{result.min_rank1_denominator:.6e}"
    )
    print(f"Results saved to: {args.output.resolve()}")

    if not args.no_plot:
        plot_result(result, config)


if __name__ == "__main__":
    main()
