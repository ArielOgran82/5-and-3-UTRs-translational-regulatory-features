# 5-and-3-UTRs-translational-regulatory-features
UTR analysis of translational regulation using classification and regression modeling

# UTR k-mer Architecture and Translational Regulation

This repository contains the complete computational analysis pipeline used to investigate whether translational regulation is encoded directly in untranslated region (UTR) sequence architecture.

The study evaluates the predictive and explanatory power of k-mer composition within:

- 5′UTRs
- 3′UTRs
- Combined 5′ + 3′ UTR architecture

using machine learning–based classification and regression frameworks.

---

## Scientific Objective

Translational control is a central layer of gene expression regulation. While many regulatory elements have been characterized, it remains unclear how much regulatory information is directly encoded in primary nucleotide composition.

This project tests whether short sequence motifs (k-mers), without additional engineered regulatory annotations, are sufficient to:

- Distinguish translational states
- Predict quantitative translational efficiency (ΔTE)
- Reveal region-specific versus integrated regulatory architectures
- Detect bimodal regulatory structure

By restricting features to k-mer composition, this analysis isolates the intrinsic regulatory signal embedded in UTR sequence architecture.

---

## Contents

- Reproducible Jupyter notebook containing the full analysis workflow
- Input dataset(s) used for modeling
- Machine learning implementation (classification and regression)
- k-mer feature extraction pipeline
- Environment specification for reproducibility

---

## Reproducibility

All analyses are fully reproducible.  
Dependencies are listed in `requirements.txt` (or `environment.yml`).  
The notebook can be executed sequentially to regenerate all results and figures.

---

## Intended Audience

This repository is intended for researchers in:

- Translational regulation
- RNA biology
- Computational genomics
- Machine learning applications in molecular biology

---

## Citation

If using this code or dataset, please cite the associated manuscript.
