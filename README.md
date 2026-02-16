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

- `mol_cell_main_version_17.9.25.ipynb` — Full reproducible analysis pipeline  
- `TIFs_PP2.csv` — Dataset used for modeling  
- `requirements.txt` — Python dependencies  
- `README.md` — Project description  

---

## Reproducibility

All analyses can be reproduced by executing:

`mol_cell_main_version_17.9.25.ipynb`

To recreate the computational environment:

```bash
pip install -r requirements.txt
```

---

## Code Availability

This repository contains the complete computational workflow used to generate the results reported in the associated manuscript.

A versioned release of this repository has been archived via Zenodo and assigned a DOI to ensure long-term preservation and citation.

---

## Conceptual Summary

This study investigates how primary sequence architecture encodes translational regulation.

By modeling k-mer composition and regulatory features, we test whether:

- Regulatory information is embedded directly within UTR nucleotide patterns  
- Translational states reflect encoded motif grammar  
- Coordinated 5′–3′ architecture defines regulatory identity  

The results provide a framework linking nucleotide-level sequence logic to translational phenotypes.
