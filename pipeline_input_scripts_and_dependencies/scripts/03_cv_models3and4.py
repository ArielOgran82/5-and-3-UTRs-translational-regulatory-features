"""
Model 3 (5utr_3utr_PP, pure k-mer classifier) and Model 4 (bi-modal:
regulatory + k-mer classifier) - genuine held-out CV performance across all
10 splits, plus a freshly-trained Model 4 (split 0, canonical) since its
saved pickles need sklearn 1.2.2 (incompatible with this env's 1.5.1).

Split scheme verified from the source notebook's own training loop for these
two model families: test_size=0.4 then 0.5 (stratified on y_categories),
matching Figure 6's convention family (NOT Model 2's 0.05/0.0075 regression
split).
"""
import pickle
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer  # not KNNImputer: negligible NaN rate here, and KNNImputer retains the full train matrix (hundreds of MB) - impractical to publish
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal

warnings.filterwarnings("ignore")

REPO = "/Users/arielo/Claude_main_folder/5-and-3-UTRs-translational-regulatory-features"
SCRATCH = f"{REPO}/_build_scratch"
RF_DIR_M3 = "/Users/arielo/Claude_main_folder/TIFseq2_main_Claude_code/RF_model_objects/5utr_3utr_PP"
RANDOM_SEEDS = [42, 145, 201, 305, 402, 78, 299, 532, 688, 745]
N_QUANTILES = [0, 0.25, 0.75, 1]
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


def feat_sel_by_stats(X_train_n_scaled, y_train, feat_col, criteria):
    feat_col_sel, feat_col_sel_pval = [], []
    for col in range(len(feat_col)):
        g0 = X_train_n_scaled[:, col][y_train == 0]
        g1 = X_train_n_scaled[:, col][y_train == 1]
        g2 = X_train_n_scaled[:, col][y_train == 2]
        if np.std(g0) != 0 or np.std(g1) != 0 or np.std(g2) != 0:
            stat, p_value = kruskal(g0, g1, g2)
            if p_value < criteria:
                feat_col_sel.append(feat_col[col])
                feat_col_sel_pval.append(p_value)
    return feat_col_sel, feat_col_sel_pval


def load_base_data():
    DF_PP = pd.read_csv(f"{REPO}/TIFs_PP2.csv")
    DF_PP = DF_PP[pd.isnull(DF_PP["seq3UTR"]) == 0]
    DF_PP = DF_PP[pd.isnull(DF_PP["seq5UTR"]) == 0]
    DF_PP = DF_PP.rename(columns={DF_PP.columns[0]: "TIF_name"})

    X = DF_PP.drop(["totalCov", "Tx_length", "TE", "pval", "TE2", "pval2", "TE3", "pval3", "TIF_len",
                     "d1_5prime", "d2_3prime", "anno", "anno2", "TIF3.relpos", "TIF5.relpos",
                     "MFE3_dotBracket", "MFE5_dotBracket",
                     "htsMean_1_10_end3", "htsMean_2_10_end3", "htsMean_end3_10_end3", "htsMean_4_10_end3",
                     "htsMean_7_10_end3", "htsMean_8_10_end3", "htsMean_9_10_end3",
                     "htsMean_1_10_end5", "htsMean_2_10_end5", "htsMean_3_10_end5", "htsMean_4_10_end5",
                     "htsMean_end5_10_end5", "htsMean_6_10_end5", "htsMean_7_10_end5", "htsMean_8_10_end5",
                     "htsMean_9_10_end5", "htsMean_10_10_end5"], axis=1)
    X = X.drop(["A_3", "T_3", "G_3", "C_3", "A_5", "T_5", "G_5", "C_5"], axis=1)
    X = X.drop(["uATG_Inframe_3", "uATG_Inframe_5", "uATG_Outframe_3", "uATG_Outframe_5",
                "uATG_Outframe_1_3", "uATG_Outframe_1_5", "uATG_Outframe_2_3", "uATG_Outframe_2_5",
                "MFE5", "MFE3"], axis=1)
    X = X.drop(["AUC_end5", "AUC_end3", "tisu_3utr", "ensemble_diversity_end5", "ensemble_diversity_end3",
                "peaks_overmedian_end5", "peaks_overmedian_end3", "peaks_undermedian_end5",
                "peaks_undermedian_end3", "peak_mean_hgt_end5", "peak_mean_hgt_end3"], axis=1)
    X = X.drop(["first_nt_3", "first_nt_5", "last_nt_3", "last_nt_5", "startMinus6", "startMinus5",
                "startMinus4", "startMinus3", "startMinus2", "startMinus1", "startPlus1"], axis=1)
    X = X.rename(columns={"rel.hts1_end3_end3": "rel.hts1_3_end3",
                           "rel.hts2_end3_end3": "rel.hts2_3_end3",
                           "rel.hts3_end3_end3": "rel.hts3_3_end3"})
    X["GC_3"] = 1 - X["GC_3"]
    X = X.rename(columns={"GC_3": "AT_3"})
    X["GC_5"] = 1 - X["GC_5"]
    X = X.rename(columns={"GC_5": "AT_5"})

    y = DF_PP["TE3"]
    y_quantiles = np.quantile(y, N_QUANTILES)
    y_categories = np.zeros(len(y))
    category = 0
    for n_quantile in range(1, len(y_quantiles)):
        temp_cat = (y > y_quantiles[n_quantile - 1]) & (y <= y_quantiles[n_quantile])
        y_categories += temp_cat.values * category
        category += 1

    X_n = X.select_dtypes(include=[np.number, "float64", "int64"]).copy()
    return DF_PP, X_n, y, y_categories


def split_60_20_20(X, y_categories, seed):
    idx = np.arange(len(X))
    idx_train, idx_rest, ycat_train, ycat_rest = train_test_split(
        idx, y_categories, test_size=0.4, shuffle=True, stratify=y_categories, random_state=seed)
    idx_test, idx_val, ycat_test, ycat_val = train_test_split(
        idx_rest, ycat_rest, test_size=0.5, shuffle=True, stratify=ycat_rest, random_state=seed)
    return idx_train, idx_val, idx_test


def main():
    DF_PP, X_n, y, y_categories = load_base_data()
    print(f"Regulatory numeric features: {X_n.shape}")
    print("Generating full hexamer x window matrix (both UTRs)...")
    parts = [generate_kmers_byRange(DF_PP[seq_col], frm, to, 6, label)
             for label, (seq_col, frm, to) in WINDOWS.items()]
    X_kmer_full = pd.concat(parts, axis=1)
    X_kmer_full.index = X_n.index
    print(f"Full k-mer matrix: {X_kmer_full.shape}")

    results_m3, results_m4 = [], []
    model4_split0 = None

    for split in range(10):
        seed = RANDOM_SEEDS[split]
        idx_train, idx_val, idx_test = split_60_20_20(X_kmer_full, y_categories, seed)
        y_train, y_val, y_test = y_categories[idx_train], y_categories[idx_val], y_categories[idx_test]

        # ---- Model 3: pure k-mer, reuse the already-trained/saved model ----
        with open(f"{RF_DIR_M3}/rf_reg_kmer_split{split}.pkl", "rb") as f:
            rf3 = pickle.load(f)
        with open(f"{RF_DIR_M3}/sel_feat_by_stats_reg_kmer_split{split}.pkl", "rb") as f:
            sel_feat3 = pickle.load(f)

        X_train_kmer_only = X_kmer_full.iloc[idx_train]
        X_test_kmer_only = X_kmer_full.iloc[idx_test]
        scaler3 = StandardScaler().fit(X_train_kmer_only.values)
        X_train3_scaled = scaler3.transform(X_train_kmer_only.values)
        X_test3_scaled = scaler3.transform(X_test_kmer_only.values)
        imputer3 = SimpleImputer(strategy="median").fit(X_train3_scaled)
        X_test3_imputed = imputer3.transform(X_test3_scaled)

        feat_idx3 = [list(X_kmer_full.columns).index(f) for f in sel_feat3]
        X_test3_sel = X_test3_imputed[:, feat_idx3]

        y_pred3 = rf3.predict(X_test3_sel)
        y_proba3 = rf3.predict_proba(X_test3_sel)
        acc3 = accuracy_score(y_test, y_pred3)
        roc3 = roc_auc_score(y_test, y_proba3, multi_class="ovo", average="macro")
        f13 = f1_score(y_test, y_pred3, average="weighted")
        results_m3.append(dict(split=split, accuracy=acc3, roc_auc=roc3, f1=f13, n_features=len(sel_feat3)))
        print(f"[Model 3 / split {split}] held-out acc={acc3:.4f} roc={roc3:.4f} f1={f13:.4f}")

        # ---- Model 4: bi-modal (regulatory + k-mer), retrain fresh ----
        X_n_train = X_n.iloc[idx_train].values
        X_n_test = X_n.iloc[idx_test].values
        X_kmer_train = X_kmer_full.iloc[idx_train].values
        X_kmer_test = X_kmer_full.iloc[idx_test].values
        feat_col4 = list(X_n.columns) + list(X_kmer_full.columns)

        X_train4_raw = np.concatenate([X_n_train, X_kmer_train], axis=1)
        X_test4_raw = np.concatenate([X_n_test, X_kmer_test], axis=1)
        scaler4 = StandardScaler().fit(X_train4_raw)
        X_train4_scaled = scaler4.transform(X_train4_raw)
        X_test4_scaled = scaler4.transform(X_test4_raw)
        imputer4 = SimpleImputer(strategy="median").fit(X_train4_scaled)
        X_train4_imputed = imputer4.transform(X_train4_scaled)
        X_test4_imputed = imputer4.transform(X_test4_scaled)

        sel_feat4, sel_feat4_pval = feat_sel_by_stats(X_train4_imputed, y_train, feat_col4, 5e-01)
        feat_idx4 = [feat_col4.index(f) for f in sel_feat4]
        X_train4_sel = X_train4_imputed[:, feat_idx4]
        X_test4_sel = X_test4_imputed[:, feat_idx4]

        rf4 = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        rf4.fit(X_train4_sel, y_train)
        y_pred4 = rf4.predict(X_test4_sel)
        y_proba4 = rf4.predict_proba(X_test4_sel)
        acc4 = accuracy_score(y_test, y_pred4)
        roc4 = roc_auc_score(y_test, y_proba4, multi_class="ovo", average="macro")
        f14 = f1_score(y_test, y_pred4, average="weighted")
        results_m4.append(dict(split=split, accuracy=acc4, roc_auc=roc4, f1=f14, n_features=len(sel_feat4)))
        print(f"[Model 4 / split {split}] held-out acc={acc4:.4f} roc={roc4:.4f} f1={f14:.4f}  (n_feat={len(sel_feat4)})")

        if split == 0:
            model4_split0 = dict(model=rf4, selected_features=sel_feat4, selected_features_pval=sel_feat4_pval,
                                  x_scaler=scaler4, imputer=imputer4, split_index=0, random_seed=seed,
                                  metrics=dict(accuracy=acc4, roc_auc=roc4, f1=f14))

    df3 = pd.DataFrame(results_m3)
    df4 = pd.DataFrame(results_m4)
    print("\n=== Model 3 (pure k-mer) CV summary ===")
    print(df3.describe().loc[["mean", "std", "min", "max"]])
    print("\n=== Model 4 (bi-modal) CV summary ===")
    print(df4.describe().loc[["mean", "std", "min", "max"]])

    df3.to_csv(f"{SCRATCH}/model3_cv_performance.csv", index=False)
    df4.to_csv(f"{SCRATCH}/model4_cv_performance.csv", index=False)
    with open(f"{SCRATCH}/model4_bimodal_split0.pkl", "wb") as f:
        pickle.dump(model4_split0, f)
    print("\nSaved model3_cv_performance.csv, model4_cv_performance.csv, model4_bimodal_split0.pkl")


if __name__ == "__main__":
    main()
