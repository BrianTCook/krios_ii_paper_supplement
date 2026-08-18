# Supplementary Repository for Cook et al., 2026: "Modeling Globular Cluster Stellar Streams with a Basis-Expansion N-body Code"

This repository provides a pathway to analyzing data presented in Cook et al., 2026. This includes example data in a folder titled "snapshot_data", a supporting Python script with associated data-loading functions, and a Jupyter notebook that shows how to plot the data. More specifically, the notebook displays stream models produced by KRIOS (a new basis-expansion N-body code introduced in Tep et al., 2025), NBODY6++GPU (Wang et al., 2015), and particle spray (Chen et al., 2025) as implemented in gala (Price-Whelan, 2022). These models are plotted on the sky plane in the great-circle reference frame (see https://gala.adrian.pw/en/latest/coordinates/greatcircle.html for more details) and in integrals-of-motion space (i.e., each star's energy and z-component of the angular momentum with respect to the host potential). The model_comparison.pdf file displays this notebook after it has been succesfully run.

Each subfolder in snapshot_data contains the last snapshot at 5 Gyr for each simulation. (The KRIOS and N-body folders also contain additional information, e.g. the Lagrange radii as a function of time.) Please contact the corresponding author (Email: btcook [at] unc.edu) for additional snapshots and supporting data from this paper.

This repository also includes a .yml file, which can be used to set up a Conda environment that helps execute the provided notebook in its entirety. To create and use this environment, run the following commands in the terminal:

conda create --name krios_env --file environment.yml
conda activate krios_env

The arXiv posting for this paper can be found at https://arxiv.org/abs/2602.13385.