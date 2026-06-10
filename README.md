# Bayesian Non-Parametric Inference for Lévy Measures in State-Space Models

[![arXiv](https://img.shields.io/badge/arXiv-2505.22587-b31b1b.svg)](https://arxiv.org/abs/2505.22587)
[![Bayesian Analysis](https://img.shields.io/badge/Bayesian%20Analysis-2025-2c3e50.svg)](https://projecteuclid.org/journals/bayesian-analysis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

Reference implementation accompanying:

> **Lin, B. Z. & Godsill, S. (2025).**
> *Bayesian Non-Parametric Inference for Lévy Measures in State-Space Models.*
> Bayesian Analysis (to appear). arXiv:2505.22587 — <https://arxiv.org/abs/2505.22587>

## Overview

Lévy processes model complex dynamics with skewness, heavy tails and
discontinuities, but inferring their Lévy measures is hard: most have intractable
likelihoods, and many exhibit infinite activity (infinitely many jumps in any
finite interval). Existing non-parametric methods further assume the Lévy process
is observed *directly*.

This repository implements the framework of the paper, which performs Bayesian
non-parametric inference of the Lévy measures of **subordinators** and **normal
variance-mean (NVM)** processes that drive a **linear state-space model**, under
discrete, noisy and possibly partial observations. The key ingredients are:

- **The Independent Gamma-Scaled Dirichlet Process (IGSDP)** — a flexible random
  measure used as the prior on the subordinator Lévy measure. The classical
  **Gamma process is recovered as a special case**, for which the paper derives
  **conjugate hyper-parameter inference**. The NVM Lévy measure is then inferred
  as an **IGSDP mixture**.
- **An identifiable parameterization of the NVM process** via an explicit
  characterization of the parameter contour, resolving the identifiability issues
  of NVM families such as the Generalized Hyperbolic process.
- **An augmented MCMC algorithm** that jointly samples the latent subordinator
  shot-noise series, the Lévy measures, and the system states/parameters using
  shot-noise (series) representations of the Lévy state-space model.

The code reproduces all **synthetic** experiments in the paper. The §5.3 results
on high-frequency tick-level FX data are not included, as they rely on
third-party (TrueFX) market data.

---

## Repository layout

```text
.
├── Simulation_Preliminaries.ipynb           # §2 tutorial: series & shot-noise representations
├── NVM_Process_Experiments.ipynb            # §5.1 NVM process Lévy-measure inference
├── Langevin_Model_Experiments.ipynb         # §5.2 Langevin state-space model inference
│
├── Levy_Generators.py                       # Lévy jump (shot-noise) generators
├── Levy_State_Space.py                      # linear Lévy state-space models
├── Filters.py                               # Kalman filter / RTS smoother + marginal likelihood
├── Common_Tools.py                          # math/plotting helpers & MCMC diagnostics
├── ground_truths.py                         # closed-form & Monte-Carlo ground-truth measures
├── posteriors.py                            # IGSDP conditional posteriors & block updates
├── mcmc_sampler.py                          # augmented MCMC samplers
├── mcmc_sampler_sigmaw2_conditional.py      # MCMC samplers in the identifiable sigma_w^2 parameterization
├── bimodal_experiments_sampler.py           # extra samplers for the bimodal experiments
│
├── TS_driven_NVM.npz                        # simulated data: tempered-stable-driven NVM
├── bimodal_driven_NVM.npz                   # simulated data: bimodal-subordinator-driven NVM
│
├── requirements.txt
└── LICENSE
```

### A note on data files

The small `*.npz` files committed to this repository (`TS_driven_NVM.npz`,
`bimodal_driven_NVM.npz`) are the **exact simulated datasets** used in the paper,
so that figures are reproduced against the same data.

The **posterior sample archives** produced by the MCMC samplers (e.g.
`langevin_ts_sigmaw2_conditional.npz`) are several gigabytes each and are *not*
tracked — they are listed in `.gitignore`. Running the corresponding notebook
cells regenerates them locally.

---

## Installation

```bash
git clone https://github.com/zhl24/To_Share_BNP_Code_Base.git
cd To_Share_BNP_Code_Base

python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate

pip install -r requirements.txt
```

The pinned versions in `requirements.txt` correspond to Python 3.12.

## Usage

Launch Jupyter and run a notebook top-to-bottom (**Restart & Run All**):

```bash
jupyter lab
```

Suggested order, following the paper:

| Notebook | Paper | Contents |
|---|---|---|
| `Simulation_Preliminaries.ipynb` | §2 | Series and generalized shot-noise representations for subordinators, NVM processes, and the Lévy state-space model. |
| `NVM_Process_Experiments.ipynb` | §5.1 | Non-parametric inference of the NVM Lévy measure from direct observations: tempered-stable-driven, its noisy variant, and a bimodal subordinator. |
| `Langevin_Model_Experiments.ipynb` | §5.2 | Full state-space inference for a linear Langevin model driven by an NVM process and observed in Gaussian noise (tempered-stable and bimodal cases). |

> **Runtime note.** The MCMC experiments are computationally intensive and were
> run on an Apple M4 MacBook Pro. Generating the full posterior archives can take
> hours and tens of gigabytes of disk; reduce the burn-in / number of iterations
> in the relevant cells for a quicker pass.

---

## Citation

If you use this toolkit, please cite:

```bibtex
@article{Lin_Godsill_2025,
  title      = {{B}ayesian {N}on-{P}arametric {I}nference for {L}\'evy {M}easures in {S}tate-{S}pace {M}odels},
  author     = {Lin, Bill Z. and Godsill, Simon},
  year       = {2025},
  journal    = {Bayesian Analysis},
  note       = {To appear; arXiv:2505.22587 [stat.ME]},
  eprint     = {2505.22587},
  eprinttype = {arXiv},
  url        = {https://arxiv.org/abs/2505.22587},
  doi        = {10.48550/arXiv.2505.22587}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE) for the full text.
