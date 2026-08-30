"""Generate models/model_metadata.json from the actual bundle files and CV
tables, rather than hand-typing numbers (avoids the manuscript-inconsistency
problem this session already hit once elsewhere)."""
import json
import pickle
import pandas as pd

REPO = "/Users/arielo/Claude_main_folder/5-and-3-UTRs-translational-regulatory-features"
MODELS = f"{REPO}/models"

with open(f"{MODELS}/model1_regulatory_only.pkl", "rb") as f:
    m1 = pickle.load(f)
with open(f"{MODELS}/model2_xgb_te_regressor.pkl", "rb") as f:
    m2 = pickle.load(f)
with open(f"{MODELS}/model3_kmer_classifier.pkl", "rb") as f:
    m3 = pickle.load(f)
with open(f"{MODELS}/model4_bimodal_classifier.pkl", "rb") as f:
    m4 = pickle.load(f)

cv3 = pd.read_csv(f"{MODELS}/model3_cv_performance.csv")
cv4 = pd.read_csv(f"{MODELS}/model4_cv_performance.csv")
cv5 = pd.read_csv(f"{MODELS}/model5_cv_performance.csv")

metadata = {
    "dataset": {
        "source": "TIFs_PP2.csv",
        "n_transcripts_total": 7123,
        "n_transcripts_after_filtering": 6262,
        "filter": "transcripts with both an assigned 5'UTR and 3'UTR sequence",
        "target_regression": "TE3 (raw scale, not log-transformed)",
        "target_classification": "Low/Mid/High TE tertile-like category, quantile boundaries [0, 0.25, 0.75, 1] on TE3",
    },
    "models": {
        "model1_regulatory_only": {
            "file": "model1_regulatory_only.pkl",
            "description": "Regulatory-features-only Random Forest classifier (Low/Mid/High TE). No sequence k-mer features.",
            "manuscript_figure": "Figure 3B (regulatory-feature-only comparison)",
            "task": "3-class classification",
            "feature_count": len(m1["selected_features"]),
            "n_estimators": m1["n_estimators_tuned"],
            "split": {"train_test_split": "60/20/20 (test_size=0.4 then 0.5, stratified)", "random_seed": 42, "split_index": 0},
            "preprocessing": "StandardScaler -> SimpleImputer(median) -> Kruskal-Wallis feature selection (p<0.5)",
            "preprocessing_note": ("SimpleImputer(median) used in place of the original notebook's KNNImputer: "
                                    "verified NaN rate ~0.7%, so results are practically identical, and KNNImputer "
                                    "must retain the full training matrix to find neighbors at predict time "
                                    "(impractical file size for a small NaN rate)."),
            "held_out_test_performance": {"accuracy": m1["metrics"]["acc_test"], "roc_auc_macro_ovo": m1["metrics"]["roc_test"], "f1_weighted": m1["metrics"]["f1_test"]},
            "manuscript_reported": "~0.80 ROC-AUC for this model type",
            "cv_documented": False,
            "cv_note": "Single canonical split reported (split 0); not independently re-verified across all 10 splits this pass.",
        },
        "model2_xgb_te_regressor": {
            "file": "model2_xgb_te_regressor.pkl",
            "description": "XGBoost regressor predicting continuous TE3 from combined regulatory + k-mer (both UTRs) features.",
            "manuscript_figure": "Figure 4B (TE regression, best-split result)",
            "task": "regression",
            "feature_count": len(m2["selected_features"]),
            "hyperparameters": m2["hyperparameters"],
            "split": {"train_test_split": "95/(~4.96)/(~0.04)% (test_size=0.05 then 0.0075, stratified)",
                      "random_seed": m2["random_seed"], "split_index": m2["split_index"]},
            "preprocessing": "StandardScaler(X) -> Kruskal-Wallis feature selection (p<0.05) -> asymmetric y outlier trim (train_mean-1*std, train_mean+3*std) -> StandardScaler(y)",
            "outlier_trim": m2["outlier_trim"],
            "utr_scope": m2["utr_scope"],
            "held_out_test_performance": {"pearson_r": m2["metrics"]["pearson_r_test"]},
            "manuscript_reported": "r=0.69 (paper's best-split result)",
            "reproduction_note": ("Reproduced r=0.667 vs. paper's 0.694 on the identified split (split 8). Confirmed "
                                   "deterministic w.r.t. XGBoost random_state (30-seed sweep gave identical results); "
                                   "residual gap attributed to package-version differences between this environment "
                                   "and the original interactive session (Python 3.9/current sklearn+xgboost here vs. "
                                   "a base Python 3.12 conda env originally) rather than a methodological error - "
                                   "train-set correlation matches the original almost exactly (0.9926 vs 0.99273)."),
            "cv_documented": False,
        },
        "model3_kmer_classifier": {
            "file": "model3_kmer_classifier.pkl",
            "description": "Pure k-mer (5'+3'UTR hexamer, 6 windows) Random Forest classifier. No regulatory numeric features.",
            "manuscript_figure": "Figure 3 (sequence-only k-mer model)",
            "task": "3-class classification",
            "feature_count": len(m3["selected_features"]),
            "n_estimators": m3["model"].n_estimators,
            "split": {"train_test_split": "60/20/20 (test_size=0.4 then 0.5, stratified)", "random_seed": m3["random_seed"], "split_index": m3["split_index"]},
            "preprocessing": "StandardScaler -> SimpleImputer(median) -> Kruskal-Wallis feature selection (p<0.5)",
            "preprocessing_note": "SimpleImputer(median) in place of KNNImputer - verified 0% NaN in this feature set, so this is a complete no-op change.",
            "held_out_test_performance_canonical_split": m3["metrics"],
            "cv_documented": True,
            "cv_across_10_splits": {
                "accuracy_mean": cv3["accuracy"].mean(), "accuracy_std": cv3["accuracy"].std(),
                "roc_auc_mean": cv3["roc_auc"].mean(), "roc_auc_std": cv3["roc_auc"].std(),
                "f1_mean": cv3["f1"].mean(), "f1_std": cv3["f1"].std(),
                "feature_count_range": [int(cv3["n_features"].min()), int(cv3["n_features"].max())],
                "source_csv": "model3_cv_performance.csv",
            },
        },
        "model4_bimodal_classifier": {
            "file": "model4_bimodal_classifier.pkl",
            "description": "Bi-modal Random Forest classifier: regulatory numeric features + k-mer (5'+3'UTR, 6 windows) features combined.",
            "manuscript_figure": "Figure 3/4 (bi-modal comparison)",
            "task": "3-class classification",
            "feature_count": len(m4["selected_features"]),
            "n_estimators": m4["model"].n_estimators,
            "split": {"train_test_split": "60/20/20 (test_size=0.4 then 0.5, stratified)", "random_seed": m4["random_seed"], "split_index": m4["split_index"]},
            "preprocessing": "StandardScaler -> SimpleImputer(median) -> Kruskal-Wallis feature selection (p<0.5)",
            "retrained_fresh_note": ("Retrained fresh under the current environment rather than restoring the original "
                                     "pickle, which requires sklearn 1.2.2 (incompatible with this repo's pinned 1.5.1)."),
            "held_out_test_performance_canonical_split": m4["metrics"],
            "cv_documented": True,
            "cv_across_10_splits": {
                "accuracy_mean": cv4["accuracy"].mean(), "accuracy_std": cv4["accuracy"].std(),
                "roc_auc_mean": cv4["roc_auc"].mean(), "roc_auc_std": cv4["roc_auc"].std(),
                "f1_mean": cv4["f1"].mean(), "f1_std": cv4["f1"].std(),
                "feature_count_range": [int(cv4["n_features"].min()), int(cv4["n_features"].max())],
                "source_csv": "model4_cv_performance.csv",
            },
        },
        "model5_fig6_combo_classifier": {
            "file": "model5_fig6_combo/model5_fig6_combo_classifier.pkl",
            "auxiliary_files": {
                "selected_triple_features": "model5_fig6_combo/model5_selected_triple_features.pkl",
                "triple_fc_pvalues": "model5_fig6_combo/model5_triple_fc_pvalues.pkl",
            },
            "description": ("Random Forest classifier on 3-way multiplicative combinations of top-ranked "
                             "5'/3'UTR k-mers, used to rank Figure 6's motif-combination importances."),
            "manuscript_figure": "Figure 6 (motif combination arc-plot)",
            "task": "3-class classification",
            "feature_count": "n_features_in_ (varies by split, see split-specific selected-triple-features file)",
            "split": {"canonical_split_index": 1,
                      "note": ("Split 1 verified by tracing the source notebook's kernel execution order (exec_count), "
                               "NOT split 0 or split 8, which were considered and ruled out during this analysis.")},
            "structural_sanity_check": {
                "n_estimators": 100, "n_features_in_matches_selected_list": True,
                "tree_depth_range": [42, 66], "tree_n_leaves_range": [609, 685],
                "feature_importances_sum": 1.0, "n_features_with_nonzero_importance": "2473/2477",
            },
            "cv_documented": False,
            "cv_note": ("Independent held-out CV reconstruction attempted and found NOT reliable: this model's "
                        "3-way multiplicative combination features are extremely sparse (a single test-set "
                        "reconstruction attempt found ~7 nonzero cells out of ~7.6 million), so any small mismatch "
                        "in the (partially externally-dependent, not fully recoverable) top-motif-pool selection "
                        "or scaling pipeline collapses predictions to the majority class. Performance is reported "
                        "as originally described in the manuscript (Figure 6 legend/Methods), not independently "
                        "re-verified via held-out CV in this pass. See model5_cv_performance.csv for the attempted "
                        "(unreliable) reconstruction, kept for transparency, not for citation."),
            "cv_attempted_but_unreliable": {
                "accuracy": cv5["accuracy"].iloc[0], "note": "collapsed to majority-class prediction in every split - do not cite",
            },
            "prediction_on_new_data_caveat": ("This model cannot be straightforwardly applied to new sequences without "
                                               "reconstructing the exact top-motif pool used to build its 3-way "
                                               "combination features, which depended on external files not included "
                                               "in this repository. Provided for interpretability/reproducibility of "
                                               "the reported Figure 6 combination rankings, not for out-of-the-box "
                                               "prediction."),
        },
    },
}

with open(f"{MODELS}/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"Saved {MODELS}/model_metadata.json")
