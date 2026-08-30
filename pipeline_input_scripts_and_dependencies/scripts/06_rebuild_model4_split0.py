"""Rebuild just Model 4's split-0 bundle with SimpleImputer (not the full
10-split CV loop, which already ran and is unaffected by this - the tiny
NaN rate makes the imputer choice immaterial to the reported metrics)."""
import sys
sys.path.insert(0, "/Users/arielo/Claude_main_folder/5-and-3-UTRs-translational-regulatory-features/_build_scratch")
import importlib
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

m = importlib.import_module("03_cv_models3and4")
SCRATCH = m.SCRATCH

DF_PP, X_n, y, y_categories = m.load_base_data()
parts = [m.generate_kmers_byRange(DF_PP[seq_col], frm, to, 6, label)
         for label, (seq_col, frm, to) in m.WINDOWS.items()]
X_kmer_full = pd.concat(parts, axis=1)
X_kmer_full.index = X_n.index

split = 0
seed = m.RANDOM_SEEDS[split]
idx_train, idx_val, idx_test = m.split_60_20_20(X_kmer_full, y_categories, seed)
y_train, y_test = y_categories[idx_train], y_categories[idx_test]

X_n_train, X_n_test = X_n.iloc[idx_train].values, X_n.iloc[idx_test].values
X_kmer_train, X_kmer_test = X_kmer_full.iloc[idx_train].values, X_kmer_full.iloc[idx_test].values
feat_col4 = list(X_n.columns) + list(X_kmer_full.columns)

X_train4_raw = np.concatenate([X_n_train, X_kmer_train], axis=1)
X_test4_raw = np.concatenate([X_n_test, X_kmer_test], axis=1)
scaler4 = StandardScaler().fit(X_train4_raw)
X_train4_scaled = scaler4.transform(X_train4_raw)
X_test4_scaled = scaler4.transform(X_test4_raw)
imputer4 = SimpleImputer(strategy="median").fit(X_train4_scaled)
X_train4_imputed = imputer4.transform(X_train4_scaled)
X_test4_imputed = imputer4.transform(X_test4_scaled)

sel_feat4, sel_feat4_pval = m.feat_sel_by_stats(X_train4_imputed, y_train, feat_col4, 5e-01)
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
print(f"Model 4 split 0 (SimpleImputer): acc={acc4:.4f} roc={roc4:.4f} f1={f14:.4f} (n_feat={len(sel_feat4)})")
print("(compare to model4_cv_performance.csv split-0 row: acc=0.6558 roc=0.8291 f1=0.6366)")

model4_split0 = dict(model=rf4, selected_features=sel_feat4, selected_features_pval=sel_feat4_pval,
                      x_scaler=scaler4, imputer=imputer4, split_index=0, random_seed=seed,
                      metrics=dict(accuracy=acc4, roc_auc=roc4, f1=f14))
with open(f"{SCRATCH}/model4_bimodal_split0.pkl", "wb") as f:
    pickle.dump(model4_split0, f)
print("Saved model4_bimodal_split0.pkl")
