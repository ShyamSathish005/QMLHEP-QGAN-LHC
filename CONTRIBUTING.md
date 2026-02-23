# Contributing to QGAN-HEP

Thanks for your interest in contributing! This project is part of the
**GSoC 2026 — Quantum Machine Learning for High Energy Physics** programme
(ML4SCI / CERN).  Contributions of all sizes are welcome.

---

## Getting started

```bash
# 1. Fork & clone
git clone https://github.com/<you>/QMLHEP-QGAN-LHC.git
cd QMLHEP-QGAN-LHC

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the training loop to check everything works
python main.py
```

---

## How to contribute

| Type | How |
|------|-----|
| **Bug report** | Open an issue with steps to reproduce and the full traceback. |
| **Feature request** | Open an issue describing the motivation and expected behaviour. |
| **Code change** | Fork → branch → PR (see below). |

### Pull-request workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-improvement
   ```
2. Make your changes.  Keep commits atomic and messages descriptive.
3. Make sure the code runs without errors:
   ```bash
   python main.py            # full training sanity check
   python synthetic_data.py  # data generation
   python physics_utils.py   # unit self-test
   ```
4. Open a Pull Request against `main`.  Describe **what** changed and **why**.

---

## Code style

* Python 3.10+.
* Use type hints for function signatures.
* Follow [PEP 8](https://peps.python.org/pep-0008/) — a formatter like
  `black` or `ruff format` is encouraged.
* Docstrings in
  [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html).

---

## Project layout

```
QMLHEP-QGAN-LHC/
├── qgan/                  # Modular QGAN package
│   ├── __init__.py
│   ├── generator.py       # Quantum generator (PennyLane)
│   ├── discriminator.py   # Classical discriminator (PyTorch)
│   ├── losses.py          # Physics-aware & MMD losses
│   ├── train.py           # Training loop
│   └── visualize.py       # Plotting & statistics
├── physics_utils.py       # Relativistic energy-momentum utilities
├── synthetic_data.py      # Double-Higgs toy data generator
├── main.py                # CLI entry point
├── qgan_hep.py            # Original monolithic script (kept for reference)
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
└── README.md
```

---

## Areas where help is especially welcome

- **Real quantum hardware runs** (IBM Quantum, IonQ, etc.).
- **Extended ansätze** — try different circuit architectures.
- **Larger physics datasets** — interface with actual LHC Open Data.
- **Automated testing** — pytest suite for the `qgan/` package.
- **CI/CD** — GitHub Actions for linting and smoke tests.

---

## Code of conduct

Be kind and constructive.  Disrespectful behaviour will not be tolerated.

---

## License

By contributing you agree that your contributions will be licensed under the
[MIT License](LICENSE).
