"""Generate synthetic Double-Higgs mass-distribution samples for testing.

Why toy data?
-------------
This script generates **simplified, on-shell 4-vectors** whose invariant-mass
spectrum peaks at the Standard-Model Higgs mass (125 GeV).  It is *not* a
full Monte-Carlo simulation — there is no parton shower, no detector response,
and no pile-up.  We use it because:

1. The official CERN Open Data (https://opendata.cern.ch) requires
   multi-GB ROOT files and experiment-specific frameworks (CMSSW, etc.)
   that are beyond the scope of this prototype.
2. For validating the QGAN architecture and the physics-aware loss, all
   we need is data that sits on the mass shell (E² = |p|² + m²) with a
   realistic momentum spread — exactly what this generator provides.
3. Replacing this with real LHC data later requires only swapping the
   data-loading call; the training pipeline is data-source agnostic.

The momentum components (px, py, pz) are drawn from zero-centred
Gaussians whose widths roughly approximate a boosted Higgs, and the
energy is computed on-shell so every event satisfies the relativistic
mass-shell constraint by construction.

Usage
-----
    python synthetic_data.py              # writes synthetic_hh_data.npz
    python synthetic_data.py --plot       # also shows a quick histogram
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def generate_double_higgs_samples(
    n_samples: int = 1000,
    m_higgs: float = 125.0,
    pt_scale: float = 60.0,
    pz_scale: float = 80.0,
    seed: int = 42,
) -> np.ndarray:
    """Return (n_samples, 4) array of [E, px, py, pz] 4-vectors.

    The momentum components are drawn from Gaussians whose widths
    roughly approximate a boosted Higgs; the energy is then computed
    on-shell:

        E = sqrt(px² + py² + pz² + m_H²)

    so every event exactly satisfies the relativistic mass-shell
    constraint by construction.

    Parameters
    ----------
    n_samples : int
        Number of events to generate.
    m_higgs : float
        Higgs mass in GeV (default 125 GeV).
    pt_scale : float
        Standard deviation for the transverse-momentum components (px, py).
    pz_scale : float
        Standard deviation for the longitudinal momentum (pz).
    seed : int
        Numpy random seed.
    """
    rng = np.random.default_rng(seed)

    px = rng.normal(0, pt_scale, n_samples)
    py = rng.normal(0, pt_scale, n_samples)
    pz = rng.normal(0, pz_scale, n_samples)

    p_sq = px**2 + py**2 + pz**2
    E = np.sqrt(p_sq + m_higgs**2)

    return np.column_stack([E, px, py, pz])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Double-Higgs 4-vector data."
    )
    parser.add_argument(
        "-n", "--n-samples", type=int, default=1000, help="Number of events"
    )
    parser.add_argument(
        "-o", "--output", default="results/synthetic_hh_data.npz",
        help="Output file (default: results/synthetic_hh_data.npz)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show invariant-mass histogram"
    )
    args = parser.parse_args(argv)

    data = generate_double_higgs_samples(n_samples=args.n_samples)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, four_vectors=data)
    print(f"[saved] {args.output}  ({args.n_samples} events)")

    # Quick sanity check
    E = data[:, 0]
    p_sq = np.sum(data[:, 1:] ** 2, axis=1)
    m_inv = np.sqrt(np.maximum(E**2 - p_sq, 0))
    print(f"  Invariant mass — mean: {m_inv.mean():.2f} GeV | "
          f"std: {m_inv.std():.4f} GeV")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].hist(m_inv, bins=50, color="steelblue", edgecolor="black")
        axes[0].axvline(125.0, color="red", ls="--", lw=2, label="125 GeV")
        axes[0].set(xlabel="Invariant mass (GeV)", ylabel="Count",
                    title="Mass distribution (should peak at 125 GeV)")
        axes[0].legend()

        pt = np.sqrt(data[:, 1]**2 + data[:, 2]**2)
        axes[1].hist(pt, bins=50, color="teal", edgecolor="black")
        axes[1].set(xlabel="pT (GeV)", ylabel="Count",
                    title="Transverse momentum")

        axes[2].hist(data[:, 0], bins=50, color="coral", edgecolor="black")
        axes[2].set(xlabel="Energy (GeV)", ylabel="Count",
                    title="Energy distribution")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
