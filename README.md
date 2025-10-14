# SCREAM: Single-cell Clustering using Representation Autoencoder of Multiomics
<img src="docs/source/_static/SCREAM_logo.png" align="right" alt="SCREAM logo" width="180" />

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/CABSEL/SCREAM/blob/main/LICENSE)

SCREAM is a deep learning framework for robust integration and clustering of multi-modal single-cell data. The method leverages Stacked Autoencoders (SAEs) to generate robust latent representations for each omics modality and uses Deep Embedding Clustering (DEC) to iteratively refine the fused multiomics latent space and cluster assignments.

**Preprint**: Chrysinas, P., Venkatesan, S., Patel, P.G., and Gunawan, R. (2025). SCREAM: Single-cell Clustering using Representation Autoencoder of Multiomics. *bioRxiv*, 2025.10.03.680290. [https://doi.org/10.1101/2025.10.03.680290](https://www.biorxiv.org/content/10.1101/2025.10.03.680290v2)

## Overview

Single-cell multiomics technologies provide unprecedented opportunities to study cellular heterogeneity, but integrating information across different omics modalities remains challenging due to:
- High dimensionality and sparsity
- Modality-specific noise characteristics
- Differences in dimensionality and information content across modalities

SCREAM addresses these challenges through a two-step approach:
1. **Modality-specific representation learning**: Independent training of Stacked Autoencoders for each omics modality
2. **Joint clustering**: Deep Embedding Clustering on concatenated latent representations

## Key Features

- **Superior clustering performance**: cnsistently achieves highest or near-highest ARI and NMI scores compared to state-of-the-art methods
- **Flexible architecture**: Adaptable to various single-cell multiomics datasets. Tested on SNARE-seq and CITE-seq datasets.
- **Robust multiomics fusion**: Effectively balances contributions from different omics modalities
- **Biologically meaningful embeddings**: Generates latent representations suitable for downstream analyses

## Installation

### Requirements
- Python >= 3.8
- TensorFlow >= 2.x
- scanpy
- muon
- pandas
- numpy

### Install from source

```bash
git clone https://github.com/cabsel/scream.git
cd scream
pip install -r requirements.txt
pip install -e .
