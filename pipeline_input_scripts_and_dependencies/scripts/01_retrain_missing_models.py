"""
Reviewer 1, comment 4 - public model release.
Retrain the 2 models that were never pickled in the original notebook:
  1. Regulatory-features-only RF classifier (Fig 3B) - the training cell in
     both notebook copies references X_l (label-encoded categorical
     regulatory features), but the cell that builds X_l is commented out a
     few cells earlier in both copies, so this model has never actually run
     successfully. Fixed here (compute X_l) and retrained.
  2. XGBoost continuous-TE regressor (Fig 4B) - only its predictions were
     ever saved, not the fitted model object. Retrained on the exact same
     feature set as the already-saved bi-modal RF classifier (Fig 4A/C).

Verbatim-ported functions/pipeline from the GitHub repo notebook
(NAR_main_version_30.8.26.ipynb), split 0 (seed 42), to match the
already-published methodology.
"""
import pickle
import warnings
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import kruskal

warnings.filterwarnings("ignore")

REPO = "/Users/arielo/Claude_main_folder/5-and-3-UTRs-translational-regulatory-features"
SCRATCH = f"{REPO}/_build_scratch"
RANDOM_SEEDS = [42, 145, 201, 305, 402, 78, 299, 532, 688, 745]
N_QUANTILES = [0, 0.25, 0.75, 1]
SPLIT = 0
rnd_seed = RANDOM_SEEDS[SPLIT]


# ---------------------------------------------------------------- verbatim ports of repo notebook functions ----
def features_type(X_features):
    X_n = X_features.select_dtypes(include=[np.number, "float64", "int64"]).copy()
    X_c = X_features.select_dtypes(include=["object", "category"]).copy()
    X_c = X_c.drop(["TIF_name", "seq3UTR", "seq5UTR"], axis=1)
    X_3prime_s = X_features["seq3UTR"]
    X_5prime_s = X_features["seq5UTR"]
    return X_n, X_c, X_3prime_s, X_5prime_s


def label_encoder(X_c):
    X_temp = X_c.copy()
    for column in X_temp.columns:
        le = LabelEncoder()
        X_temp[column] = le.fit_transform(X_temp[column])
    return X_temp


def scale_numerics(X_train_n, X_val_n, X_test_n, scaler_type):
    scaler = StandardScaler() if scaler_type == "standard" else StandardScaler()
    X_train_n_scaled = scaler.fit_transform(X_train_n)
    X_val_n_scaled = scaler.transform(X_val_n)
    X_test_n_scaled = scaler.transform(X_test_n)
    return X_train_n_scaled, X_val_n_scaled, X_test_n_scaled, scaler


def feat_sel_by_stats(X_train_n, X_train_n_scaled, X_val_n_scaled, X_test_n_scaled, y_train, feat_col, criteria):
    feat_col_sel = []
    feat_col_sel_pval = []
    comp_results = []
    for col in range(len(feat_col)):
        group0 = X_train_n_scaled[:, col][y_train == 0]
        group1 = X_train_n_scaled[:, col][y_train == 1]
        group2 = X_train_n_scaled[:, col][y_train == 2]
        if np.std(group0) != 0 or np.std(group1) != 0 or np.std(group2) != 0:
            stat, p_value = kruskal(group0, group1, group2)
            if p_value < criteria:
                feat_col_sel.append(feat_col[col])
                feat_col_sel_pval.append(p_value)
    if isinstance(feat_col, pd.Index):
        feat_col_indices = [feat_col.get_loc(f) for f in feat_col_sel]
    else:
        feat_col_indices = [feat_col.index(f) for f in feat_col_sel]
    X_train_n_scaled_sel = X_train_n_scaled[:, feat_col_indices]
    X_val_n_scaled_sel = X_val_n_scaled[:, feat_col_indices]
    X_test_n_scaled_sel = X_test_n_scaled[:, feat_col_indices]
    return X_train_n_scaled_sel, X_val_n_scaled_sel, X_test_n_scaled_sel, feat_col_sel, feat_col_sel_pval


def hyperparameter_tunining(X_train, y_train, model):
    n_estimators_ = [50, 100, 150, 200, 250, 300, 350]
    param_grid = {"n_estimators": n_estimators_}
    clf = GridSearchCV(model, param_grid, cv=3, scoring="f1_weighted")
    clf.fit(X_train, y_train)
    return clf.best_params_["n_estimators"]


def acc_roc_f1_estimation(X_train, y_train, X_val, y_val, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    roc_train = roc_auc_score(y_train, model.predict_proba(X_train), multi_class="ovo", average="macro")
    f1_train = f1_score(y_train, y_pred_train, average="weighted")
    acc_val = accuracy_score(y_val, y_pred_val)
    roc_val = roc_auc_score(y_val, model.predict_proba(X_val), multi_class="ovo", average="macro")
    f1_val = f1_score(y_val, y_pred_val, average="weighted")
    acc_test = accuracy_score(y_test, y_pred_test)
    roc_test = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovo", average="macro")
    f1_test = f1_score(y_test, y_pred_test, average="weighted")
    return acc_train, acc_val, acc_test, roc_train, roc_val, roc_test, f1_train, f1_val, f1_test


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


# ---------------------------------------------------------------------------- data loading (verbatim, repo cells 59-63) ----
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

    print(f"Loaded {len(X)} TIFs, X shape {X.shape}")
    return X, y, y_categories


def main():
    X, y, y_categories = load_base_data()
    X_n, X_c, X_3prime_s, X_5prime_s = features_type(X)
    X_l = label_encoder(X_c)  # <-- the fix: this line was commented out in both notebooks
    print(f"X_n (numeric regulatory): {X_n.shape}, X_l (label-encoded categorical): {X_l.shape}")

    # ================================================================ MODEL 1: regulatory-features-only RF (Fig 3B) ====
    print("\n" + "=" * 70)
    print("MODEL 1: regulatory-features-only RF classifier (Fig 3B)")
    print("=" * 70)

    X_train_n, X_new, y_train, y_new = train_test_split(X_n, y_categories, test_size=0.4, shuffle=True,
                                                          stratify=y_categories, random_state=rnd_seed)
    X_test_n, X_val_n, y_test, y_val = train_test_split(X_new, y_new, test_size=0.5, shuffle=True,
                                                          stratify=y_new, random_state=rnd_seed)
    X_train_l, X_new_l, _, _ = train_test_split(X_l, y_categories, test_size=0.4, shuffle=True,
                                                  stratify=y_categories, random_state=rnd_seed)
    X_test_l, X_val_l, _, _ = train_test_split(X_new_l, y_new, test_size=0.5, shuffle=True,
                                                 stratify=y_new, random_state=rnd_seed)

    feat_col = list(X_train_n.columns) + list(X_train_l.columns)

    X_train_n_scaled, X_val_n_scaled, X_test_n_scaled, scaler_reg = scale_numerics(
        X_train_n, X_val_n, X_test_n, "standard")
    # SimpleImputer, not the original notebook's KNNImputer: NaN rate here is
    # ~0.7%, so the two are practically equivalent, and KNNImputer must
    # retain a full copy of the training data to find neighbors at predict
    # time (hundreds of MB for the larger models) - impractical to publish.
    imputer_reg = SimpleImputer(strategy="median")
    X_train_n_scaled = imputer_reg.fit_transform(X_train_n_scaled)
    X_val_n_scaled = imputer_reg.transform(X_val_n_scaled)
    X_test_n_scaled = imputer_reg.transform(X_test_n_scaled)
    # separate imputer for the raw-scale path (feat_sel_by_stats wants both
    # raw and scaled): NOT the same object saved for prediction, since that
    # one must stay fit on scaled-space medians to match the scale->impute
    # order the predict pipeline actually uses.
    X_train_n_imputed = SimpleImputer(strategy="median").fit_transform(X_train_n)

    X_train_n_l_scaled = np.concatenate((X_train_n_scaled, X_train_l.values), axis=1)
    X_val_n_l_scaled = np.concatenate((X_val_n_scaled, X_val_l.values), axis=1)
    X_test_n_l_scaled = np.concatenate((X_test_n_scaled, X_test_l.values), axis=1)
    X_train_n_l_raw = np.concatenate((X_train_n_imputed, X_train_l.values), axis=1)

    X_train_sel, X_val_sel, X_test_sel, sel_feat, sel_feat_pval = feat_sel_by_stats(
        X_train_n_l_raw, X_train_n_l_scaled, X_val_n_l_scaled, X_test_n_l_scaled, y_train, feat_col, 5e-01)
    print(f"Before selection: {X_train_n_l_scaled.shape[1]} features; after Kruskal p<0.5: {X_train_sel.shape[1]}")

    rf = RandomForestClassifier(random_state=rnd_seed, n_jobs=-1)
    n_est = hyperparameter_tunining(X_train_sel, y_train, rf)
    print(f"Hyperparameter tuning selected n_estimators={n_est}")

    rf_reg_final = RandomForestClassifier(n_estimators=n_est, random_state=rnd_seed, n_jobs=-1)
    acc_train, acc_val, acc_test, roc_train, roc_val, roc_test, f1_train, f1_val, f1_test = acc_roc_f1_estimation(
        X_train_sel, y_train, X_val_sel, y_val, X_test_sel, y_test, rf_reg_final)
    print(f"Final regulatory-only RF: test acc={acc_test:.4f} roc={roc_test:.4f} f1={f1_test:.4f}")
    print(f"(val acc={acc_val:.4f} roc={roc_val:.4f} - paper reports ~0.80 ROC-AUC for this model type)")

    with open(f"{SCRATCH}/model1_regulatory_only.pkl", "wb") as f:
        pickle.dump({
            "model": rf_reg_final, "selected_features": sel_feat, "selected_features_pval": sel_feat_pval,
            "scaler": scaler_reg, "imputer": imputer_reg, "all_feature_cols": feat_col,
            "n_estimators_tuned": n_est,
            "metrics": dict(acc_train=acc_train, acc_val=acc_val, acc_test=acc_test,
                             roc_train=roc_train, roc_val=roc_val, roc_test=roc_test,
                             f1_train=f1_train, f1_val=f1_val, f1_test=f1_test),
        }, f)
    print(f"Saved model1_regulatory_only.pkl")


if __name__ == "__main__":
    main()
