"""
Model 5 (Fig6 combo classifier, kmer_comb_top10) - genuine held-out CV
across all 10 splits.

Sidesteps the ambiguous historical top-N-per-window motif pool (which
depends on external CSV files at ~/Desktop/utr_ai_desktop/kmers/ with an
ambiguous commented-out .head(10) toggle - not reliably reconstructible):
for each split, the already-saved sel_feat_by_stats_reg_split{i}.pkl tells
us exactly which triples survived training-time filtering. We only need the
individual motifs behind those triples, computed directly from sequences,
then reproduce the exact scaling (minmax, fit on that split's train
partition only) and triple-product construction (comb_multiply_array:
simple elementwise product of the 3 scaled columns) to evaluate the
already-trained, already-saved classifier on genuine held-out test rows.

Split scheme verified from the source notebook's "0)" combo-generation cell:
test_size=0.4 then 0.5, stratified on y_categories, random_state=RANDOM_SEEDS[n].
"""
import pickle
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

TIFPROJ = "/Users/arielo/Claude_main_folder/TIFseq2_main_Claude_code"
REPO = "/Users/arielo/Claude_main_folder/5-and-3-UTRs-translational-regulatory-features"
SCRATCH = f"{REPO}/_build_scratch"
MODEL_DIR = f"{TIFPROJ}/RF_model_objects/kmer_comb_top10"
RANDOM_SEEDS = [42, 145, 201, 305, 402, 78, 299, 532, 688, 745]
N_QUANTILES = [0, 0.25, 0.75, 1]
WINDOW_BOUNDS = {
    "5_1st": ("seq5UTR", 0, 40), "5_2nd": ("seq5UTR", -40, None),
    "3_1st": ("seq3UTR", 0, 40), "3_2nd": ("seq3UTR", 40, 80),
    "3_3rd": ("seq3UTR", -80, -40), "3_4th": ("seq3UTR", -40, None),
}


def kmer_freq_for_motifs(seqs, frm, to, kmers_needed):
    """Normalized frequency (matches generate_kmers_byRange) for a specific
    small set of k-mers in one window, across all sequences."""
    k = 6
    out = {kmer: np.zeros(len(seqs)) for kmer in kmers_needed}
    for idx, seq in enumerate(seqs):
        sub = seq[frm:to]
        if len(sub) < k:
            continue
        total = len(sub) - k + 1
        counts = {}
        for p in range(total):
            kk = sub[p:p + k]
            if kk in kmers_needed:
                counts[kk] = counts.get(kk, 0) + 1
        for kmer, c in counts.items():
            out[kmer][idx] = c / total
    return out


def main():
    df = pd.read_csv(f"{TIFPROJ}/input/TIFs_PP2.csv")
    df = df[pd.isnull(df["seq3UTR"]) == 0]
    df = df[pd.isnull(df["seq5UTR"]) == 0].reset_index(drop=True)
    y = df["TE3"].values
    y_quantiles = np.quantile(y, N_QUANTILES)
    y_categories = np.zeros(len(y))
    cat = 0
    for n in range(1, len(y_quantiles)):
        y_categories += ((y > y_quantiles[n - 1]) & (y <= y_quantiles[n])) * cat
        cat += 1
    print(f"n_total transcripts: {len(df)} (expected 6262)")

    results = []
    for split in range(10):
        seed = RANDOM_SEEDS[split]
        with open(f"{MODEL_DIR}/rf_reg_split{split}.pkl", "rb") as f:
            rf = pickle.load(f)
        with open(f"{MODEL_DIR}/sel_feat_by_stats_reg_split{split}.pkl", "rb") as f:
            sel_feat = pickle.load(f)

        # unique individual motifs behind this split's surviving triples
        triples = [t.split(",") for t in sel_feat]
        unique_motifs = sorted({m for triple in triples for m in triple})
        by_window = {}
        for m in unique_motifs:
            kmer, win = m[:6], m[7:]
            by_window.setdefault(win, set()).add(kmer)

        freq_cols = {}
        for win, kmers in by_window.items():
            seq_col, frm, to = WINDOW_BOUNDS[win]
            vals = kmer_freq_for_motifs(df[seq_col], frm, to, kmers)
            for kmer, arr in vals.items():
                freq_cols[f"{kmer}_{win}"] = arr
        X_motifs = pd.DataFrame(freq_cols, index=df.index)

        idx_all = np.arange(len(df))
        idx_train, idx_rest, ycat_train, ycat_rest = train_test_split(
            idx_all, y_categories, test_size=0.4, shuffle=True, stratify=y_categories, random_state=seed)
        idx_test, idx_val, ycat_test, ycat_val = train_test_split(
            idx_rest, ycat_rest, test_size=0.5, shuffle=True, stratify=ycat_rest, random_state=seed)

        scaler = MinMaxScaler().fit(X_motifs.iloc[idx_train].values)
        X_test_scaled = scaler.transform(X_motifs.iloc[idx_test].values)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_motifs.columns)

        X_test_combo = np.column_stack([
            X_test_scaled_df[triple[0]].values * X_test_scaled_df[triple[1]].values * X_test_scaled_df[triple[2]].values
            for triple in triples
        ])

        y_pred = rf.predict(X_test_combo)
        y_proba = rf.predict_proba(X_test_combo)
        acc = accuracy_score(ycat_test, y_pred)
        roc = roc_auc_score(ycat_test, y_proba, multi_class="ovo", average="macro")
        f1 = f1_score(ycat_test, y_pred, average="weighted")
        results.append(dict(split=split, accuracy=acc, roc_auc=roc, f1=f1,
                             n_features=len(sel_feat), n_unique_motifs=len(unique_motifs)))
        print(f"[Model 5 / split {split}] held-out acc={acc:.4f} roc={roc:.4f} f1={f1:.4f} "
              f"(n_triples={len(sel_feat)}, n_unique_motifs={len(unique_motifs)})")

    res_df = pd.DataFrame(results)
    print("\n=== Model 5 (Fig6 combo) CV summary ===")
    print(res_df.describe().loc[["mean", "std", "min", "max"]])
    res_df.to_csv(f"{SCRATCH}/model5_cv_performance.csv", index=False)
    print("\nSaved model5_cv_performance.csv")


if __name__ == "__main__":
    main()
