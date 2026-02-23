"""Entry point — train the QGAN and save results to results/."""

import json
import os

from qgan import train_qgan, visualize_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("QGAN-HEP: Quantum GAN for High-Energy Physics")
    print("=" * 70)

    generator, history = train_qgan(epochs=200, batch_size=16)

    print("\nTraining complete — generating diagnostics …\n")

    plot_path = os.path.join(RESULTS_DIR, "qgan_hep_analysis.png")
    visualize_results(generator, history, save_path=plot_path)

    # Persist scalar metrics so reviewers can inspect without re-running
    metrics = {
        "final_d_loss": history["d_loss"][-1],
        "final_g_loss": history["g_loss"][-1],
        "final_phys_loss": history["g_phys"][-1],
        "final_mmd_loss": history["g_mmd"][-1],
        "barren_plateau_detected": history["barren_plateau_detected"],
        "mode_collapse_detected": history["mode_collapse_detected"],
    }
    metrics_path = os.path.join(RESULTS_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {metrics_path}")

    print("=" * 70)
    print("Done.  All outputs are in results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
