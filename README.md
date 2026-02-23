# QGAN-HEP: Quantum GANs for High Energy Physics

**GSoC 2026 | Quantum Machine Learning for High Energy Physics (ML4SCI / CERN)**

A hybrid quantum-classical Generative Adversarial Network for simulating
LHC particle data (Double-Higgs production) using PennyLane variational
circuits and PyTorch.

---

## Why Double-Higgs?

Measuring Higgs-boson pair production (pp → HH) at the LHC is the only
direct probe of the Higgs trilinear self-coupling λ₃ — a fundamental
parameter of the Standard Model that tells us the shape of the Higgs
potential.  The problem: HH events are **extremely rare** (~1000×
rarer than single-Higgs), so Monte-Carlo simulation eats enormous
compute budgets.  A fast, physics-aware generative model that can
produce realistic HH 4-vectors on demand would directly accelerate
analysis pipelines.

This project explores whether a **quantum GAN** — whose entangling
layers provide an exponentially large Hilbert space for latent
representations — can learn the kinematics of such events while
respecting the hard constraint $E^2 = |\mathbf{p}|^2 + m^2$.

---

## Quick start

```bash
# clone & set up
git clone https://github.com/shyamsathish/QMLHEP-QGAN-LHC.git
cd QMLHEP-QGAN-LHC
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train the QGAN (≈ 2-3 min on CPU)
python main.py

# generate 1 000 synthetic Double-Higgs events
python synthetic_data.py --plot

# run the physics-utils self-test
python physics_utils.py
```

---

## Project layout

```
QMLHEP-QGAN-LHC/
├── qgan/                   # modular QGAN package
│   ├── __init__.py
│   ├── generator.py        # PennyLane quantum generator (4 qubits, 2 layers)
│   ├── discriminator.py    # classical MLP discriminator
│   ├── losses.py           # physics-aware + MMD losses
│   ├── train.py            # training loop with diagnostics
│   └── visualize.py        # 6-panel analysis & statistics
├── physics_utils.py        # relativistic energy-momentum utilities
├── synthetic_data.py       # Double-Higgs toy data generator (1 000 events)
├── main.py                 # CLI entry point
├── qgan_hep.py             # original monolithic script (kept for reference)
├── requirements.txt
├── LICENSE                  # MIT
├── CONTRIBUTING.md
└── README.md
```

---

## Architecture

```
┌─────────────────────────────────┐
│  QUANTUM GENERATOR  (4 qubits) │
│  AngleEmbedding → SEL ×2       │
│  → PauliZ meas → Linear(4→4)   │
└───────────────┬─────────────────┘
                │  4-vectors (E, px, py, pz)
     ┌──────────┴──────────┐
     ▼                     ▼
┌────────────┐    ┌────────────────┐
│Discriminator│    │ Physics loss   │
│  (MLP)     │    │ (E²−p²−m²)²   │
└────────────┘    └────────────────┘
     └──────┬──────────────┘
            ▼
   L = L_bce + 0.5·L_phys + 1.0·L_mmd
```

### Loss function

| Component | Formula | Purpose |
|-----------|---------|---------|
| Adversarial | BCE(D(G(z)), 1) | Fool the discriminator |
| Physics | mean((E² − \|p\|² − m²)²) | Mass-shell constraint |
| Distribution | MMD(fake, real) | Match real kinematics |

---

## Key features

- **Barren-plateau mitigation** — small-variance init (σ = 0.1),
  gradient-norm monitoring, hardware-efficient ansatz.
- **Mode-collapse prevention** — three-part loss ensures the generator
  cannot trivially satisfy all objectives at once.
- **Physics-aware training** — the relativistic energy-momentum loss
  (`physics_utils.py`) keeps generated events on the mass shell.
- **Synthetic data generator** — `synthetic_data.py` produces on-shell
  Double-Higgs 4-vectors (125 GeV peak) for rapid prototyping.

---

## Results (200 epochs, default config)

After running `python main.py`, outputs are saved to `results/`:

- `results/qgan_hep_analysis.png` — 6-panel diagnostic figure
- `results/training_metrics.json` — final loss values

See that folder for the actual output from the current codebase.

---

## Configuration

Tunable knobs live at the top of `qgan/generator.py` and in the
`train_qgan()` signature:

```python
# Circuit
N_QUBITS = 4          # qubits in the generator
N_LAYERS = 2          # StronglyEntanglingLayers depth

# Training  (arguments to train_qgan)
epochs      = 200
batch_size  = 16
lr          = 0.01
lambda_phys = 0.5     # weight for the physics loss
lambda_mmd  = 1.0     # weight for the MMD loss
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Gradient norm < 1e-6 for 50+ epochs | Reduce `N_LAYERS` or increase init variance |
| All generated masses collapse to one value | Increase `lambda_mmd` (1.0 → 2.0) |
| CUDA OOM | Reduce `batch_size` (16 → 8) |
| Training > 5 min on CPU | Reduce `epochs` or use GPU |

---

## References

1. Du et al. (2020) — "Barren plateaus in quantum neural network training"
2. Cerezo et al. (2021) — "Variational quantum algorithms"
3. Goodfellow et al. (2014) — "Generative Adversarial Networks"
4. Gretton et al. (2012) — "A kernel two-sample test" (MMD)
5. PennyLane documentation — <https://pennylane.ai>

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)
