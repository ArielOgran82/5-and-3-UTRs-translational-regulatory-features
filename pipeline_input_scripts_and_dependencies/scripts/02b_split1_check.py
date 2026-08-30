"""
Model 2: XGBoost continuous-TE regressor (Fig 4B).
CORRECTED to match the actual documented pipeline (repo notebook cells
55/149, "Bi-modality Regression analysis" / "For 0.05 test size") -
95%/5% train/test split, feat_sel_by_stats at p<0.05, and xgb_reg_tune with
fixed hyperparameters (not the generic xgb_reg used in the first attempt,
which gave test r=0.50 vs. the paper's reported r=0.69 - a red flag that
caught this mismatch before publishing the wrong model).
"""
import pickle
import warnings
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal, pearsonr

warnings.filterwarnings("ignore")

REPO = "/Users/arielo/Claude_main_folder/5-and-3-UTRs-translational-regulatory-features"
SCRATCH = f"{REPO}/_build_scratch"
RANDOM_SEEDS = [42, 145, 201, 305, 402, 78, 299, 532, 688, 745]
N_QUANTILES = [0, 0.25, 0.75, 1]
SPLIT = 1
rnd_seed = RANDOM_SEEDS[SPLIT]

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


def feat_sel_by_stats(X_train_n, X_train_n_scaled, X_val_n_scaled, X_test_n_scaled, y_train, feat_col, criteria):
    feat_col_sel = []
    feat_col_sel_pval = []
    for col in range(len(feat_col)):
        group0 = X_train_n_scaled[:, col][y_train == 0]
        group1 = X_train_n_scaled[:, col][y_train == 1]
        group2 = X_train_n_scaled[:, col][y_train == 2]
        if np.std(group0) != 0 or np.std(group1) != 0 or np.std(group2) != 0:
            stat, p_value = kruskal(group0, group1, group2)
            if p_value < criteria:
                feat_col_sel.append(feat_col[col])
                feat_col_sel_pval.append(p_value)
    feat_col_indices = [feat_col.index(f) for f in feat_col_sel]
    return (X_train_n_scaled[:, feat_col_indices], X_val_n_scaled[:, feat_col_indices],
            X_test_n_scaled[:, feat_col_indices], feat_col_sel, feat_col_sel_pval)


def xgb_reg_tune(X_train_, X_test_, y_train_num, y_test_num, n_estimators, learning_rate, max_depth, gamma,
                  bound_lower, bound_higher):
    scaler = StandardScaler()
    y_train_num_scaled = scaler.fit_transform(np.expand_dims(y_train_num, 1))
    y_test_num_scaled = scaler.transform(np.expand_dims(y_test_num, 1))

    keep_train = ((y_train_num_scaled > bound_lower) & (y_train_num_scaled < bound_higher)).ravel()
    y_train_num_scaled = y_train_num_scaled[keep_train]
    X_train_ = X_train_[keep_train, :]

    keep_test = ((y_test_num_scaled > bound_lower) & (y_test_num_scaled < bound_higher)).ravel()
    y_test_num_scaled = y_test_num_scaled[keep_test]
    X_test_ = X_test_[keep_test, :]

    model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                              max_depth=max_depth, gamma=gamma)
    model.fit(X_train_, y_train_num_scaled.ravel())
    y_train_pred = model.predict(X_train_)
    y_test_pred = model.predict(X_test_)

    corr_train = np.corrcoef(y_train_num_scaled[:, 0], y_train_pred)[0, 1]
    corr_test = np.corrcoef(y_test_num_scaled[:, 0], y_test_pred)[0, 1]
    return model, scaler, corr_train, corr_test


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
    y_categories = np.zeros_like(y)
    category = 0
    for n_quantile in range(1, len(y_quantiles)):
        temp_cat = (y > y_quantiles[n_quantile - 1]) & (y <= y_quantiles[n_quantile])
        y_categories += temp_cat * category
        category += 1

    X_n = X.select_dtypes(include=[np.number, "float64", "int64"]).copy()
    return DF_PP, X_n, y, y_categories


def main():
    DF_PP, X_n, y, y_categories = load_base_data()
    print(f"Regulatory numeric features: {X_n.shape}")

    print("Generating full hexamer x window matrix (both UTRs)...")
    parts = [generate_kmers_byRange(DF_PP[seq_col], frm, to, 6, label)
              for label, (seq_col, frm, to) in WINDOWS.items()]
    X_kmer_full = pd.concat(parts, axis=1)
    X_kmer_full.index = X_n.index
    print(f"Full k-mer matrix: {X_kmer_full.shape}")

    feat_col_0 = list(X_n.columns)

    # 95% train / ~4.96% test / ~0.04% val - matches paper's "95% train, 5% held out"
    X_train_n, X_new, y_train, y_new = train_test_split(
        X_n, y_categories, test_size=0.05, shuffle=True, stratify=y_categories, random_state=rnd_seed)
    X_test_n, X_val_n, y_test, y_val = train_test_split(
        X_new, y_new, test_size=0.0075, shuffle=True, stratify=y_new, random_state=rnd_seed)
    X_train_n, X_new, y_train_num, y_new_num = train_test_split(
        X_n, y, test_size=0.05, shuffle=True, stratify=y_categories, random_state=rnd_seed)
    X_test_n, X_val_n, y_test_num, y_val_num = train_test_split(
        X_new, y_new_num, test_size=0.0075, shuffle=True, stratify=y_new, random_state=rnd_seed)

    X_train_n_kmer, X_new_k, _, _ = train_test_split(
        X_kmer_full, y_categories, test_size=0.05, shuffle=True, stratify=y_categories, random_state=rnd_seed)
    X_test_n_kmer, X_val_n_kmer, _, _ = train_test_split(
        X_new_k, y_new, test_size=0.0075, shuffle=True, stratify=y_new, random_state=rnd_seed)

    feat_col_1 = list(X_train_n_kmer.columns)
    feat_col = feat_col_0 + feat_col_1

    X_train_n_arr = np.concatenate((X_train_n.values, X_train_n_kmer.values), axis=1)
    X_val_n_arr = np.concatenate((X_val_n.values, X_val_n_kmer.values), axis=1)
    X_test_n_arr = np.concatenate((X_test_n.values, X_test_n_kmer.values), axis=1)

    scaler_x = StandardScaler()
    X_train_n_scaled = scaler_x.fit_transform(X_train_n_arr)
    X_val_n_scaled = scaler_x.transform(X_val_n_arr)
    X_test_n_scaled = scaler_x.transform(X_test_n_arr)
    print(f"Combined (regulatory+kmer) matrix: train {X_train_n_scaled.shape}, test {X_test_n_scaled.shape}")

    print("\n" + "=" * 70)
    print("MODEL 2 (corrected): XGBoost continuous-TE regressor (Fig 4B)")
    print("=" * 70)

    X_train_sel, X_val_sel, X_test_sel, sel_feat, sel_feat_pval = feat_sel_by_stats(
        X_train_n_arr, X_train_n_scaled, X_val_n_scaled, X_test_n_scaled, y_train, feat_col, 5e-02)
    print(f"Before selection: {X_train_n_scaled.shape[1]} features; after Kruskal p<0.05: {len(sel_feat)}")

    xgb_model, y_scaler, corr_train, corr_test = xgb_reg_tune(
        X_train_=X_train_sel, X_test_=X_test_sel, y_train_num=y_train_num, y_test_num=y_test_num,
        n_estimators=1200, learning_rate=0.12, max_depth=8, gamma=0.05,
        bound_lower=-1.75, bound_higher=2)

    print(f"Pearson correlation - train: {corr_train:.4f}")
    print(f"Pearson correlation - test:  {corr_test:.4f}  (paper reports r=0.69)")

    with open(f"{SCRATCH}/model2_xgb_regressor.pkl", "wb") as f:
        pickle.dump({
            "model": xgb_model, "selected_features": sel_feat, "selected_features_pval": sel_feat_pval,
            "x_scaler": scaler_x, "y_scaler": y_scaler,
            "outlier_bounds_scaled_y": (-1.75, 2),
            "hyperparameters": dict(n_estimators=1200, learning_rate=0.12, max_depth=8, gamma=0.05),
            "metrics": dict(pearson_r_train=corr_train, pearson_r_test=corr_test),
        }, f)
    print("Saved model2_xgb_regressor.pkl")


if __name__ == "__main__":
    main()
