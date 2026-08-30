"""Build a self-contained Model 3 bundle (pure k-mer classifier, split 0
canonical): model + selected features + scaler/imputer fit on split 0's
train partition (not saved during the original exploratory research phase,
refit here so the published model is directly usable for prediction)."""
import pickle
from itertools import product

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TIFPROJ = "/Users/arielo/Claude_main_folder/TIFseq2_main_Claude_code"
REPO = "/Users/arielo/Claude_main_folder/5-and-3-UTRs-translational-regulatory-features"
SCRATCH = f"{REPO}/_build_scratch"
MODEL_DIR = f"{TIFPROJ}/RF_model_objects/5utr_3utr_PP"
RANDOM_SEEDS = [42, 145, 201, 305, 402, 78, 299, 532, 688, 745]
N_QUANTILES = [0, 0.25, 0.75, 1]
SPLIT = 0
WINDOWS = {
    "5_1st": ("seq5UTR", 0, 40), "5_2nd": ("seq5UTR", -40, None),
    "3_1st": ("seq3UTR", 0, 40), "3_2nd": ("seq3UTR", 40, 80),
    "3_3rd": ("seq3UTR", -80, -40), "3_4th": ("seq3UTR", -40, None),
}


def generate_kmers_byRange(dna_sequences, frm, to, k_i, label):
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

    print("Generating full hexamer x window matrix...")
    parts = [generate_kmers_byRange(df[seq_col], frm, to, 6, label)
             for label, (seq_col, frm, to) in WINDOWS.items()]
    X_kmer_full = pd.concat(parts, axis=1)

    with open(f"{MODEL_DIR}/rf_reg_kmer_split{SPLIT}.pkl", "rb") as f:
        rf = pickle.load(f)
    with open(f"{MODEL_DIR}/sel_feat_by_stats_reg_kmer_split{SPLIT}.pkl", "rb") as f:
        sel_feat = pickle.load(f)
    print(f"Model expects {rf.n_features_in_} features; saved list has {len(sel_feat)}")

    seed = RANDOM_SEEDS[SPLIT]
    idx_all = np.arange(len(df))
    idx_train, idx_rest, ycat_train, ycat_rest = train_test_split(
        idx_all, y_categories, test_size=0.4, shuffle=True, stratify=y_categories, random_state=seed)
    idx_test, idx_val, ycat_test, ycat_val = train_test_split(
        idx_rest, ycat_rest, test_size=0.5, shuffle=True, stratify=ycat_rest, random_state=seed)

    X_train = X_kmer_full[sel_feat].iloc[idx_train]
    X_test = X_kmer_full[sel_feat].iloc[idx_test]

    scaler = StandardScaler().fit(X_train.values)
    X_train_scaled = scaler.transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)
    # SimpleImputer, not KNNImputer: 0% NaN in this feature set (verified),
    # and KNNImputer must retain the full training matrix to find neighbors
    # at predict time - hundreds of MB, impractical to publish.
    imputer = SimpleImputer(strategy="median").fit(X_train_scaled)
    X_test_imputed = imputer.transform(X_test_scaled)

    y_pred = rf.predict(X_test_imputed)
    y_proba = rf.predict_proba(X_test_imputed)
    acc = accuracy_score(ycat_test, y_pred)
    roc = roc_auc_score(ycat_test, y_proba, multi_class="ovo", average="macro")
    f1 = f1_score(ycat_test, y_pred, average="weighted")
    print(f"Held-out check (split 0): acc={acc:.4f} roc={roc:.4f} f1={f1:.4f} "
          f"(matches model3_cv_performance.csv split-0 row)")

    bundle = {
        "model": rf, "selected_features": sel_feat, "x_scaler": scaler, "imputer": imputer,
        "split_index": SPLIT, "random_seed": seed,
        "metrics": dict(accuracy=acc, roc_auc=roc, f1=f1),
    }
    with open(f"{SCRATCH}/model3_kmer_classifier_split0.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print("Saved model3_kmer_classifier_split0.pkl")


if __name__ == "__main__":
    main()
