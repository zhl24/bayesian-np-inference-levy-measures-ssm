# Toolkit for Bayesian Non-Parametric Inference for Lévy Measures in State-Space Models

> **This repository contains the code and data required to reproduce all 
> experiments in our paper on simulated data (experiments in section 6.3 on FX data not included)**:  
>
> > **Lin B.Z. & Godsill S. (2025)**  
> > *Bayesian Non-Parametric Inference for Lévy Measures in State-Space Models.*  
> > arXiv:2505.22587 · https://arxiv.org/abs/2505.22587  
>
> >
> In addition to reproducibility, the repository **teaches the essential
> preliminaries for our methods**—see the `Simulation_Preliminaries.ipynb`.

> If you are only interested in using the methods, feel free to fork or cherry-pick.  
> **If you wish to verify the results in the paper, clone this repo and run the notebooks as instructed below.**


## 📂 Repository layout

```text
.
├── Simulation_Preliminaries.ipynb        # method preliminaries
├── Simulated_Data_Experiments.ipynb      # all simulated-data experiments
├── Simulated_Experiment_Data.npz         # exact data used in the paper (≈400 kB)
│
├── Common_Tools.py                       # plotting & math helpers
├── Filters.py                            # Kalman filter + marginal likelihood
├── Levy_Generators.py                    # Lévy series generators
├── Levy_State_Space.py                   # Lévy SSM via shot-noise reps
├── ground_truths.py                      # ground-truth simulators
├── posteriors.py                         # posterior diagnostics
├── mcmc_sampler.py                       # main MCMC algorithm
│
├── requirements.txt
└── LICENSE

---

## Quick start

```bash
git clone https://github.com/zhl24/To_Share_BNP_Code_Base.git
cd To_Share_BNP_Code_Base
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/Simulation_Preliminaries.ipynb

Each notebook can be executed top-to-bottom with Restart & Run All and
reproduces the results reported in the paper.

⸻

All experiments run on an M4 Macbook Pro
⸻

Development setup

pip install -r requirements-dev.txt  

⸻

Citation

If you use this toolkit, please cite the following article:

@article{Lin_Godsill_2025,
  title   = {{B}ayesian {N}on-{P}arametric {I}nference for {L}\'evy {M}easures in {S}tate-{S}pace {M}odels},
  author  = {Lin, B. Z. and Godsill, S.},
  year    = {2025},
  month   = {May},
  eprint  = {2505.22587},
  eprinttype = {arXiv},
  url     = {https://arxiv.org/abs/2505.22587},
  doi     = {10.48550/arXiv.2505.22587},
  note    = {arXiv:2505.22587 [stat]},
  publisher = {arXiv}
}

⸻

License

Released under the MIT License – see the LICENSE file for full
text.

Copy-paste → **Commit new file** → done.  
Your README now points users straight to your paper for citation.
