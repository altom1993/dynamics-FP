"""Reproduce manuscript Fig. 3: energy dynamics for four ramp strengths.

Curves:
    beta/beta_c1 = 0.2, 1, 4
    beta/beta_c2 = 1

The program uses the optimized trapezoidal/Hadamard solver in
``fermi_polaron_trapezoid_optimized.py``.  The numerical parameters match the original plotting script: kF=1, n_q=100, dt=1/100, n_t=12000, n_seed=7.

The short-time contact and energy coefficients are generated automatically from the general short-time chi_q expansion, so no beta-specific hard-coded polynomial is needed.  Completed curves are cached and reused on later runs.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc

import matplotlib.pyplot as plt
import numpy as np

from fermi_polaron_trapezoid_optimized import (
    SolverConfig,
    build_short_time_coefficients,
    run_simulation,
)


plt.rc('text', usetex=True)


BETA_C1 = 2.0 * np.sqrt(2.0 / np.pi)
BETA_C2 = 2.0 / np.sqrt(np.pi)

# Internal identifiers and the corresponding physical beta values.
CURVES = {
    "0p2_bc1": 0.2 * BETA_C1,
    "1_bc1": BETA_C1,
    "4_bc1": 4.0 * BETA_C1,
    "1_bc2": BETA_C2,
}


def short_time_series_coefficients(
    config: SolverConfig,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return coefficients of C(t) and E(t)/E_F through the displayed order.

    Write

        chi_q(t) = A*t^(1/2) + b*q^2*t^(3/2)
                   + c*q^4*t^(5/2) + ... .

    Then

        C(t) = C1*t + C2*t^2 + C3*t^3 + ...

    follows by analytically performing the radial q integral.  Tan's sweep
    relation is then integrated term by term to obtain E(t)/E_F.  These
    formulas reproduce the supplied beta=0.2*beta_c1 and beta=beta_c1
    polynomials, and also give the required beta=4*beta_c1 and beta=beta_c2
    matching conditions.
    """

    # q=0 and q=kF are sufficient to extract A, b, and c because cx3 is proportional to q^2 and cx5 is proportional to q^4.
    q_probe = np.array([0.0, config.kf])
    coeff = build_short_time_coefficients(config, q_probe)

    amplitude_a = coeff.cx1[0]
    amplitude_b = coeff.cx3[1] / config.kf**2
    amplitude_c = coeff.cx5[1] / config.kf**4

    kf = config.kf
    contact_c1 = abs(amplitude_a) ** 2 * kf**3 / (6.0 * np.pi**2)
    contact_c2 = (
        2.0
        * np.real(np.conj(amplitude_a) * amplitude_b)
        * kf**5
        / (10.0 * np.pi**2)
    )
    contact_c3 = (
        (abs(amplitude_b) ** 2 + 2.0 * np.real(np.conj(amplitude_a) * amplitude_c))
        * kf**7
        / (14.0 * np.pi**2)
    )
    contact_coefficients = np.array([contact_c1, contact_c2, contact_c3])

    # dE/dt = C/(8*pi*beta*t^(3/2)).  Division by E_F=kF^2/2 converts the result to the dimensionless E/E_F plotted in the manuscript.
    energy_scale = 2.0 / kf**2
    energy_coefficients = energy_scale * np.array(
        [
            contact_c1 / (4.0 * np.pi * config.beta),
            contact_c2 / (12.0 * np.pi * config.beta),
            contact_c3 / (20.0 * np.pi * config.beta),
        ]
    )
    return contact_coefficients, energy_coefficients


def short_time_energy(t: np.ndarray, config: SolverConfig) -> np.ndarray:
    """General short-time E(t)/E_F through order t^(5/2)."""

    _, coefficients = short_time_series_coefficients(config)
    return (
        coefficients[0] * t**0.5
        + coefficients[1] * t**1.5
        + coefficients[2] * t**2.5
    )


def energy_from_contact(
    time_grid: np.ndarray,
    contact: np.ndarray,
    config: SolverConfig,
) -> np.ndarray:
    """Integrate Tan's sweep relation with a second-order trapezoidal rule."""

    energy = np.empty_like(contact, dtype=float)
    energy[: config.n_seed] = short_time_energy(
        time_grid[: config.n_seed], config
    )

    # Dimensional rate followed by conversion to E/E_F.
    rate = contact / (8.0 * np.pi * config.beta * time_grid**1.5)
    energy_scale = 2.0 / config.kf**2
    increments = (
        energy_scale
        * 0.5
        * config.dt
        * (rate[config.n_seed - 1 : -1] + rate[config.n_seed :])
    )
    energy[config.n_seed :] = (
        energy[config.n_seed - 1] + np.cumsum(increments)
    )
    return energy


def cache_filename(curve_id: str) -> Path:
    return Path(f"figure3_{curve_id}_cache.npz")


def cache_matches(data, config: SolverConfig) -> bool:
    required = {"time", "contact", "energy", "beta", "dt", "n_q", "n_t", "n_seed"}
    if not required.issubset(data.files):
        return False
    return (
        np.isclose(float(data["beta"]), config.beta)
        and np.isclose(float(data["dt"]), config.dt)
        and int(data["n_q"]) == config.n_q
        and int(data["n_t"]) == config.n_t
        and int(data["n_seed"]) == config.n_seed
    )


def figure2_contact_cache(curve_id: str) -> Path | None:
    """Return a compatible Fig. 2 cache name for the first three curves."""

    names = {
        "0p2_bc1": Path("figure2_beta_0p2_cache.npz"),
        "1_bc1": Path("figure2_beta_1p0_cache.npz"),
        "4_bc1": Path("figure2_beta_4p0_cache.npz"),
    }
    return names.get(curve_id)


def contact_cache_matches(data, config: SolverConfig) -> bool:
    required = {"time", "contact", "beta", "dt", "n_q", "n_t", "n_seed"}
    if not required.issubset(data.files):
        return False
    return (
        np.isclose(float(data["beta"]), config.beta)
        and np.isclose(float(data["dt"]), config.dt)
        and int(data["n_q"]) == config.n_q
        and int(data["n_t"]) == config.n_t
        and int(data["n_seed"]) == config.n_seed
    )


def obtain_energy_curve(
    curve_id: str,
    beta: float,
    base_config: SolverConfig,
    recompute: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a valid cache or calculate an energy curve."""

    config = SolverConfig(
        beta=beta,
        kf=base_config.kf,
        n_q=base_config.n_q,
        dt=base_config.dt,
        n_t=base_config.n_t,
        n_seed=base_config.n_seed,
        show_progress=base_config.show_progress,
    )
    cache = cache_filename(curve_id)

    if cache.exists() and not recompute:
        with np.load(cache) as data:
            if cache_matches(data, config):
                print(f"Loading cached curve: {cache}")
                return data["time"].copy(), data["energy"].copy()

    # If Fig. 2 was already calculated with identical parameters, reuse its contact curve and perform only the inexpensive Tan-sweep integration.
    contact_cache = figure2_contact_cache(curve_id)
    time_grid = None
    contact = None
    if contact_cache is not None and contact_cache.exists() and not recompute:
        with np.load(contact_cache) as data:
            if contact_cache_matches(data, config):
                print(f"Reusing contact from: {contact_cache}")
                time_grid = data["time"].copy()
                contact = data["contact"].copy()

    if contact is None:
        print(f"Calculating energy curve: {curve_id}")
        result = run_simulation(config)
        time_grid = result.time.copy()
        contact = result.contact.copy()
        print(
            "  minimum |SM denominator| = "
            f"{result.min_rank1_denominator:.6e}"
        )
        del result
        gc.collect()

    energy = energy_from_contact(time_grid, contact, config)
    np.savez_compressed(
        cache,
        time=time_grid,
        contact=contact,
        energy=energy,
        beta=config.beta,
        dt=config.dt,
        n_q=config.n_q,
        n_t=config.n_t,
        n_seed=config.n_seed,
    )
    print(f"Saved cache: {cache}")
    return time_grid, energy


def draw_figure3(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    png_output: Path,
) -> None:
    """Draw and save the logarithmic-time energy figure."""

    fig, axis = plt.subplots(figsize=(3.375, 3.0))

    for spine in axis.spines.values():
        spine.set_linewidth(1.2)
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        labelsize=12,
        width=1.2,
        top=True,
        right=True,
    )
    axis.minorticks_on()
    axis.tick_params(
        axis="both",
        which="minor",
        direction="in",
        width=0.9,
        top=True,
        right=True,
        labelsize=0,
    )

    styles = {
        "0p2_bc1": dict(
            label=r"$\beta/\beta_{c1}=0.2$", linestyle="-.", color="blue"
        ),
        "1_bc1": dict(
            label=r"$\beta/\beta_{c1}=1$", linestyle="-", color="red"
        ),
        "4_bc1": dict(
            label=r"$\beta/\beta_{c1}=4$", linestyle="--", color="black"
        ),
        "1_bc2": dict(
            label=r"$\beta/\beta_{c2}=1$",
            linestyle=(0, (0.8, 0.8)),
            color="green",
        ),
    }

    for curve_id in CURVES:
        time_grid, energy = curves[curve_id]
        # With kF=m=1, epsilon_F=1/2 and t_F=2.
        axis.plot(time_grid / 2.0, energy, linewidth=1.2, **styles[curve_id])

    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.55, .52),
        fontsize=9,
        frameon=False,
        handlelength=1.8,
    )
    # axis.set_xscale("log")
    axis.set_xlim(0.0, 40.0)
    axis.set_ylim(0.0, 1.5)
    axis.set_xlabel(r"$t/t_F$", fontsize=13)
    axis.set_ylabel(r"$E/E_F$", fontsize=13)

    # Transparent backgrounds are convenient for manuscript composition.
    fig.patch.set_alpha(0.0)
    axis.set_facecolor("none")
    fig.savefig(png_output, dpi=600, bbox_inches="tight", transparent=True)
    print(f"Saved: {png_output.resolve()}")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--png", type=Path, default=Path("fermi_polaron_figure3.png")
    )
    args, _unknown = parser.parse_known_args()  # compatible with Jupyter
    return args


def main() -> None:
    args = parse_args()
    if args.quick_test:
        base_config = SolverConfig(
            n_q=20,
            dt=1.0 / 100.0,
            n_t=160,
            n_seed=7,
            show_progress=False,
        )
    else:
        base_config = SolverConfig(
            n_q=100,
            dt=1.0 / 100.0,
            n_t=12_000,
            n_seed=7,
        )

    curves = {
        curve_id: obtain_energy_curve(
            curve_id, beta, base_config, args.recompute
        )
        for curve_id, beta in CURVES.items()
    }
    draw_figure3(curves, args.png)


if __name__ == "__main__":
    main()

