# Toolkit for Bayesian Non-Parametric Inference for Lévy Measures in State-Space Models

> **This repository contains the code and data required to reproduce all 
> experiments in our paper on simulated data (section 6.3 not included)**:  
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


## Repository layout

.
├── Simulation_Preliminaries.ipynb      # Preliminaries for our methods
├── Simulated_Data_Experiments.ipynb    # All Experiments in our paper for simulated data
├──
├── src/levy_bnp/                           # core generators & MCMC kernels
├── Simulated_Experiment_Data.npz           # Exact data used in the paper (≈400 kB)
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

Notebooks at a glance

Notebook	Purpose	Typical runtime*
Simulation_Preliminaries.ipynb	single tempered-stable path, seed checks	< 1 min
01_single_path.ipynb	reproduces Fig. 2 sample path (paper)	2 – 3 min
02_parameter_inference.ipynb	full Langevin LM inference, reproduces Table 1	~10 min

* M1 MacBook; see notebook headers for per-cell timings.

⸻

Development setup (optional)

pip install -r requirements-dev.txt   # pytest, black, nbstripout …
pytest -q                             # run tests
pre-commit install                    # strip notebook output on commit


⸻

Citation

If you use this toolkit, please cite the following article:

@article{Lin_Godsill_2025,
  title   = {Bayesian Non-Parametric Inference for L{\'e}vy Measures in State-Space Models},
  author  = {Lin, Bill Z. and Godsill, Simon},
  year    = {2025},
  month   = {May},
  eprint  = {2505.22587},
  eprinttype = {arXiv},
  url     = {https://arxiv.org/abs/2505.22587},
  doi     = {10.48550/arXiv.2505.22587},
  note    = {arXiv:2505.22587 [stat]},
  publisher = {arXiv}
}

A CITATION.cff file will be provided for automatic GitHub/Zotero export.

⸻

License

Released under the MIT License – see the LICENSE file for full
text.

Copy-paste → **Commit new file** → done.  
Your README now points users straight to your paper for citation.
