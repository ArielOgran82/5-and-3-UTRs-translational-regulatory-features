# UTR Sequence Architecture Encodes Translational Regulatory States

## Overview

This repository contains the complete computational workflow used to identify regulatory features and cis-regulatory elements encoded within untranslated region (UTR) sequence architecture that contribute to translational control.

We systematically test whether short nucleotide motifs (k-mers) and regulatory features derived from:

- 5′ UTR sequences  
- 3′ UTR sequences  
- Combined 5′ + 3′ UTR architecture  

are sufficient to identify and rank sequence-derived regulatory features according to their relative contribution to translational regulation.

The analysis is implemented as a fully reproducible Jupyter notebook.

---

## Scientific Rationale

Cis-regulatory elements within 5′ and 3′ untranslated regions (UTRs) are critical for controlling mRNA translation efficiency (TE), yet their coordinated functions remain poorly understood due to their complexity and variability.

Here, we combined polysome profiling with TIF-seq2 to determine the translational efficiency (TE) of transcript isoforms (TIFs) with precisely defined 5′ and 3′ ends. We extracted regulatory features, positional k-mers, secondary structure characteristics, and TE measurements to construct a multifaceted input space for machine learning (ML) modeling.

Together, these findings expand the understanding of translation regulation and suggest that UTR regulatory features evolve primarily to fine-tune an inherently robust translation machinery to achieve balanced protein expression.

---

## Analytical Framework

### 1. Regulatory Feature and k-mer Extraction

We extracted a comprehensive set of sequence-derived regulatory features from 5′ and 3′ UTRs to construct the machine learning input space.

This includes:

- Enumeration of all possible k-mers (4^k)
- Frequency-based k-mer feature matrix construction
- Positional k-mer distribution metrics
- Identification of enriched and depleted sequence motifs
- Secondary structure–related sequence characteristics
- AU-rich and polyadenylation-associated elements
- Region-specific feature derivation (5′UTR, 3′UTR, and combined architectures)

These features collectively capture both compositional and positional regulatory information embedded within UTR sequence architecture.

### 2. Classification Analysis

Models were trained independently on:

- 5′UTR k-mers  
- 3′UTR k-mers  
- Combined UTR k-mers  

**Objective:**  
Evaluate whether sequence composition and regulatory features distinguish translational states.

### 3. Regression Modeling

Using gradient-boosted regression (XGBoost):

- Predict continuous translational metrics (e.g., ΔTE)
- Capture nonlinear motif interactions
- Assess region-specific versus integrated regulatory architecture
- Rank regulatory features by relative importance

### 4. Bi-modality Analysis

We evaluated whether translational output reflects:

- Continuous modulation  
- Or discrete regulatory regimes encoded in sequence composition  

This analysis tests whether translational control operates as a graded continuum or as structured regulatory states.

---

## Repository Contents

- `NAR_main_version_30.8.26.ipynb` — Full reproducible analysis pipeline
- `TIFs_PP2.csv` — Dataset used for modeling
- `models/` — Trained models, ready to load and use (see below)
- `examples/predict_new_transcript.ipynb` — Runnable example: load each model and generate a prediction
- `pipeline_input_scripts_and_dependencies/` — Scripts documenting exactly how each trained model was built and validated
- `requirements.txt` / `environment.yml` — Python dependencies (pip and conda)
- `LICENSE` — MIT license
- `CITATION.cff` — Citation metadata
- `README.md` — Project description

---

## Trained Models

Five trained models are provided in `models/`, covering the regulatory-only,
pure k-mer, bi-modal, XGBoost regression, and Figure-6-combination model
families described in the manuscript. Each model's exact feature set,
preprocessing pipeline, held-out performance, and known limitations are
documented in `models/model_metadata.json` — read this before using any
model, since they are **not interchangeable in what input they require**
(some need only a raw sequence; others need a full precomputed feature row
from the TIF-seq2 pipeline). `examples/predict_new_transcript.ipynb`
demonstrates correct usage of every model.

---

## Reproducibility

All analyses can be reproduced by executing:

`NAR_main_version_30.8.26.ipynb`

To recreate the computational environment:

```bash
pip install -r requirements.txt
# or: conda env create -f environment.yml
```

To use the trained models without retraining, see
`examples/predict_new_transcript.ipynb`.

---

## Code Availability

This repository contains the complete computational workflow, the trained
models, and a runnable usage example needed to reproduce and reuse the
results reported in the associated manuscript.

A versioned release of this repository is archived via Zenodo, providing a
persistent DOI for long-term preservation and citation (see `CITATION.cff`
for the DOI once assigned).

---

## Conceptual Summary

This study investigates how primary sequence architecture encodes translational regulation.

By modeling k-mer composition and regulatory features, we test whether:

- Regulatory information is embedded directly within UTR nucleotide patterns  
- Translational states reflect encoded motif grammar  
- Coordinated 5′–3′ architecture defines regulatory identity  

The results provide a framework linking nucleotide-level sequence logic to translational phenotypes.
