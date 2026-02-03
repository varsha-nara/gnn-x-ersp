# Interpretable Prototype-based Graph Information Bottleneck
The official source code for Interpretable Prototype-based Graph Information Bottleneck at NeurIPS 2023 (https://github.com/sang-woo-seo/PGIB). The official code for the SPMotif dataset is at (https://github.com/Wuyxin/DIR-GNN/tree/main). This repository runs the PGIB code on the SPMotif dataset with bias 0.3 for CPU-only configuration.

## Steps to run PGIB on SPMotif-0.3
1. Clone this repository and install conda if necessary.
2. Download the SPMotif dataset from this notebook (https://colab.research.google.com/drive/1XMCQQTHg91dhWCYyHz5vduql86j1hHhk?authuser=1#scrollTo=fMX3LVayVS7Q).
3. Ensure the dataset is placed under datasets/SPMotif-0.3/raw/
4. Create a conda environment with python 3.9 and activate it.
```
conda create -n pgib-spmotif python=3.9 -y
conda activate pgib-spmotif
```
5. Install the following requirements in this order.
```
python -m pip install --upgrade pip
pip install torch==1.11.0 torchvision torchaudio
conda install "numpy<2.0" -y
pip install -f https://data.pyg.org/whl/torch-1.11.0+cpu.html torch-scatter==2.0.9 torch-sparse==0.6.13
pip install torch-geometric==2.0.4
pip install networkx rdkit matplotlib
```
6. Run
```
python -m models.train_gnns
```
7. Ensure to deactivate the environment after use.
```
conda deactivate
```
