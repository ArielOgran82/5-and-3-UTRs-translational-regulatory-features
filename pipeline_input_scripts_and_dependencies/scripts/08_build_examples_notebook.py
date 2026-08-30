import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell

REPO = "/Users/arielo/Claude_main_folder/5-and-3-UTRs-translational-regulatory-features"
NB_PATH = f"{REPO}/examples/predict_new_transcript.ipynb"

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

md(r"""
# Predicting translation efficiency for a transcript

This notebook demonstrates how to load and use the five trained models in
`../models/`. **Read this before using any model** - they are not
interchangeable in what input they need:

| Model | Needs | Works on a brand-new sequence you just typed in? |
|---|---|---|
| **Model 3** (pure k-mer classifier) | 5'UTR + 3'UTR sequence only | **Yes** |
| Model 1 (regulatory-only classifier) | a full regulatory feature row | No - see below |
| Model 2 (XGBoost TE regressor) | a full regulatory feature row + k-mers | No - see below |
| Model 4 (bi-modal classifier) | a full regulatory feature row + k-mers | No - see below |
| Model 5 (Fig6 combo classifier) | — | Not supported for new-data prediction at all |

**Why Models 1, 2, and 4 can't run on sequence alone:** roughly half of
their "regulatory" features (`capProxPeakHeights_end3`, `rel.hts1_3_end3`,
`pltu_total_end3`, and similar) are derived from actual TIF-seq2 sequencing
coverage at that specific transcript's observed 5'/3' ends - they describe
*where reads piled up*, not something you can compute from the DNA letters
alone. To use these models you need a transcript that has already been
through the TIF-seq2 processing pipeline (i.e., already has a row in a
`TIFs_PP2.csv`-format table). Section 2 below demonstrates this using an
existing dataset row, clearly labeled as such - it is a pipeline
demonstration, not a novel-sequence prediction.

**Why Model 5 isn't demonstrated for prediction:** its 3-way combination
features depend on a specific top-motif pool that isn't fully recoverable
from this repository (see `../models/model_metadata.json`). It's included
for interpreting Figure 6's already-reported motif rankings, not for
predicting on new data.

See `../models/model_metadata.json` for full provenance, feature counts,
held-out performance, and known limitations for every model.
""")

md("## Setup")
code(r"""
import pickle
import json
from itertools import product

import numpy as np
import pandas as pd

REPO = ".."  # run this notebook from within examples/
MODELS = f"{REPO}/models"

with open(f"{MODELS}/model_metadata.json") as f:
    METADATA = json.load(f)


def generate_kmers_byRange(dna_sequences, frm, to, k_i, label):
    # Hexamer frequency (count / possible positions) within [frm:to] of each
    # sequence. Verbatim from the original analysis pipeline.
    kmers = ["".join(p) for p in product("ACGT", repeat=k_i)]
    kmer_frequencies = {kmer: [] for kmer in kmers}
    for sequence in dna_sequences:
        subseq = sequence[frm:to]
        seq_len = len(subseq)
        if seq_len == 0 or seq_len < k_i:
            for kmer in kmers:
                kmer_frequencies[kmer].append(0)
            continue
        sequence_counts = {kmer: 0 for kmer in kmers}
        for i in range(seq_len - k_i + 1):
            kmer = subseq[i:i + k_i]
            if kmer in sequence_counts:
                sequence_counts[kmer] += 1
        total_kmers = seq_len - k_i + 1
        for kmer in kmers:
            kmer_frequencies[kmer].append(sequence_counts[kmer] / total_kmers)
    df = pd.DataFrame(kmer_frequencies, index=dna_sequences.index)
    df.columns = [f"{col}_{label}" for col in df.columns]
    return df


WINDOWS = {
    "5_1st": ("seq5UTR", 0, 40), "5_2nd": ("seq5UTR", -40, None),
    "3_1st": ("seq3UTR", 0, 40), "3_2nd": ("seq3UTR", 40, 80),
    "3_3rd": ("seq3UTR", -80, -40), "3_4th": ("seq3UTR", -40, None),
}


def build_kmer_matrix(seq5utr, seq3utr):
    # Build the full 6-window hexamer matrix for one or more transcripts.
    # seq5utr, seq3utr: pandas Series of sequences, same index.
    seqs = {"seq5UTR": seq5utr, "seq3UTR": seq3utr}
    parts = [generate_kmers_byRange(seqs[col], frm, to, 6, label)
             for label, (col, frm, to) in WINDOWS.items()]
    return pd.concat(parts, axis=1)


def build_regulatory_features(df_rows):
    # Reproduce the exact column drops + renames used to build the 52-column
    # regulatory feature set (load_base_data() in the model-build scripts) -
    # required for a raw TIFs_PP2.csv row to line up with what Models 1/2/4
    # were trained on. Only numeric columns are kept, matching training time.
    X = df_rows.drop(columns=["totalCov", "Tx_length", "TE", "pval", "TE2", "pval2", "TE3", "pval3", "TIF_len",
                     "d1_5prime", "d2_3prime", "anno", "anno2", "TIF3.relpos", "TIF5.relpos",
                     "MFE3_dotBracket", "MFE5_dotBracket",
                     "htsMean_1_10_end3", "htsMean_2_10_end3", "htsMean_end3_10_end3", "htsMean_4_10_end3",
                     "htsMean_7_10_end3", "htsMean_8_10_end3", "htsMean_9_10_end3",
                     "htsMean_1_10_end5", "htsMean_2_10_end5", "htsMean_3_10_end5", "htsMean_4_10_end5",
                     "htsMean_end5_10_end5", "htsMean_6_10_end5", "htsMean_7_10_end5", "htsMean_8_10_end5",
                     "htsMean_9_10_end5", "htsMean_10_10_end5"], errors="ignore")
    X = X.drop(columns=["A_3", "T_3", "G_3", "C_3", "A_5", "T_5", "G_5", "C_5"], errors="ignore")
    X = X.drop(columns=["uATG_Inframe_3", "uATG_Inframe_5", "uATG_Outframe_3", "uATG_Outframe_5",
                "uATG_Outframe_1_3", "uATG_Outframe_1_5", "uATG_Outframe_2_3", "uATG_Outframe_2_5",
                "MFE5", "MFE3"], errors="ignore")
    X = X.drop(columns=["AUC_end5", "AUC_end3", "tisu_3utr", "ensemble_diversity_end5", "ensemble_diversity_end3",
                "peaks_overmedian_end5", "peaks_overmedian_end3", "peaks_undermedian_end5",
                "peaks_undermedian_end3", "peak_mean_hgt_end5", "peak_mean_hgt_end3"], errors="ignore")
    X = X.drop(columns=["first_nt_3", "first_nt_5", "last_nt_3", "last_nt_5", "startMinus6", "startMinus5",
                "startMinus4", "startMinus3", "startMinus2", "startMinus1", "startPlus1"], errors="ignore")
    X = X.rename(columns={"rel.hts1_end3_end3": "rel.hts1_3_end3",
                           "rel.hts2_end3_end3": "rel.hts2_3_end3",
                           "rel.hts3_end3_end3": "rel.hts3_3_end3"})
    if "GC_3" in X.columns:
        X["GC_3"] = 1 - X["GC_3"]
        X = X.rename(columns={"GC_3": "AT_3"})
    if "GC_5" in X.columns:
        X["GC_5"] = 1 - X["GC_5"]
        X = X.rename(columns={"GC_5": "AT_5"})
    return X.select_dtypes(include=[np.number])
""")

md(r"""
## 1. Model 3 - pure k-mer classifier (works on any new sequence)

Paste in your own 5'UTR and 3'UTR sequence below. This model only looks at
hexamer content across the six 40-nt windows - no coverage data required.
""")
code(r"""
example_5utr = "GCTAGCTAGCTAGCATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
example_3utr = "TTTCCGGTTCCGGCATCGATCGATCGATCGAAATAAACGATCGATCGATCGATCGATCGATCGATCGATC"

seq5 = pd.Series([example_5utr], name="seq5UTR")
seq3 = pd.Series([example_3utr], name="seq3UTR")
X_kmer = build_kmer_matrix(seq5, seq3)

with open(f"{MODELS}/model3_kmer_classifier.pkl", "rb") as f:
    m3 = pickle.load(f)

X_sel = X_kmer[m3["selected_features"]]
X_scaled = m3["x_scaler"].transform(X_sel.values)
X_imputed = m3["imputer"].transform(X_scaled)

pred = m3["model"].predict(X_imputed)
proba = m3["model"].predict_proba(X_imputed)

category_names = {0: "Low TE", 1: "Mid TE", 2: "High TE"}
print(f"Predicted category: {category_names[int(pred[0])]}")
print(f"Class probabilities (Low / Mid / High): {np.round(proba[0], 3)}")
""")

md(r"""
## 2. Models 1, 2, 4 - pipeline demonstration on an existing transcript

These models need the full regulatory feature set, which requires actual
TIF-seq2 coverage data. This section loads one transcript already present in
`TIFs_PP2.csv` and runs it through each model's exact preprocessing
pipeline - **this demonstrates correct usage of the saved model artifacts,
it is not a prediction on a novel sequence.**
""")
code(r"""
df = pd.read_csv(f"{REPO}/TIFs_PP2.csv")
df = df[pd.isnull(df["seq3UTR"]) == 0]
df = df[pd.isnull(df["seq5UTR"]) == 0].reset_index(drop=True)

example_row = df.iloc[[0]]  # first valid transcript, just for demonstration
print(f"Demonstration transcript: {example_row.iloc[0, 0]}")

X_kmer_example = build_kmer_matrix(example_row["seq5UTR"], example_row["seq3UTR"])
X_kmer_example.index = example_row.index
""")

code(r"""
X_reg_full = build_regulatory_features(example_row)

# --- Model 1: regulatory-only classifier ---
with open(f"{MODELS}/model1_regulatory_only.pkl", "rb") as f:
    m1 = pickle.load(f)

X_reg1 = X_reg_full.reindex(columns=m1["all_feature_cols"])
X_reg1_scaled = m1["scaler"].transform(X_reg1.values)
X_reg1_imputed = m1["imputer"].transform(X_reg1_scaled)
feat_idx1 = [m1["all_feature_cols"].index(f) for f in m1["selected_features"]]
X_reg1_sel = X_reg1_imputed[:, feat_idx1]

pred1 = m1["model"].predict(X_reg1_sel)
proba1 = m1["model"].predict_proba(X_reg1_sel)
print(f"Model 1 predicted category: {category_names[int(pred1[0])]}, probabilities: {np.round(proba1[0], 3)}")
""")

code(r"""
# --- Model 4: bi-modal classifier (regulatory + k-mer) ---
with open(f"{MODELS}/model4_bimodal_classifier.pkl", "rb") as f:
    m4 = pickle.load(f)

# feature order at training time was: all regulatory numeric columns, then
# the full 6-window k-mer matrix, concatenated in that order (see
# _build_scratch/03_cv_models3and4.py) - selected_features indexes into
# that combined name list, so we rebuild the same combined frame here.
reg_numeric_cols = [c for c in X_reg_full.select_dtypes(include=[np.number]).columns]
X_combined4 = pd.concat([X_reg_full[reg_numeric_cols].reset_index(drop=True),
                          X_kmer_example.reset_index(drop=True)], axis=1)
X_combined4 = X_combined4.reindex(columns=list(dict.fromkeys(reg_numeric_cols + list(X_kmer_example.columns))))

available4 = [f for f in m4["selected_features"] if f in X_combined4.columns]
if len(available4) < len(m4["selected_features"]):
    print(f"Note: {len(m4['selected_features']) - len(available4)}/{len(m4['selected_features'])} "
          f"selected features not present in this row's schema - results below use only the available ones "
          f"and are for pipeline demonstration, not a citable prediction.")

X4_full_scaled = m4["x_scaler"].transform(X_combined4.reindex(columns=reg_numeric_cols + list(X_kmer_example.columns)).values)
X4_imputed = m4["imputer"].transform(X4_full_scaled)
feat_idx4 = [(reg_numeric_cols + list(X_kmer_example.columns)).index(f) for f in m4["selected_features"]]
X4_sel = X4_imputed[:, feat_idx4]

pred4 = m4["model"].predict(X4_sel)
proba4 = m4["model"].predict_proba(X4_sel)
print(f"Model 4 predicted category: {category_names[int(pred4[0])]}, probabilities: {np.round(proba4[0], 3)}")
""")

md(r"""
Model 2 (the XGBoost TE regressor) follows the same pattern as Model 4
above - concatenate the regulatory numeric columns with the full k-mer
matrix in that order, scale with `x_scaler`, select `selected_features`,
then call `model.predict(...)` and inverse-transform with `y_scaler` to get
back to the raw TE3 scale. Omitted here to keep this notebook short; see
`model_metadata.json` for its exact feature/hyperparameter record.
""")

md(r"""
## 3. Model 5 - interpretation only, not demonstrated for prediction

Model 5 ranks Figure 6's 3-way motif combinations by importance. Use it by
loading `model5_fig6_combo/model5_fig6_combo_classifier.pkl` and inspecting
`.feature_importances_` alongside `model5_selected_triple_features.pkl`
(the exact triple names in matching order) - this reproduces the importance
ranking behind Figure 6 directly, without needing to run new predictions.
See `../models/model_metadata.json` for the full caveat.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "rf_rescue", "language": "python", "name": "rf_rescue"}}
nbf.write(nb, NB_PATH)
print(f"Wrote {len(cells)} cells to {NB_PATH}")
