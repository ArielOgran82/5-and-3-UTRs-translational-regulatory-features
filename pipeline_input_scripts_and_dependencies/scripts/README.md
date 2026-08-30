# Model-building and validation scripts

These scripts document the exact methodology used to produce the trained
models in `../../models/` and the held-out cross-validation numbers reported
in `../../models/model_metadata.json`. They are provided for **methodological
transparency** — read them to see precisely how each model was built,
validated, and what design decisions were made along the way (including
decisions logged in code comments, e.g. why `SimpleImputer` replaces the
original notebook's `KNNImputer`, or why Model 2 uses split 8 specifically).

## Important: these scripts are not a one-click reproduction harness

Several scripts read from an original research directory structure (raw
per-split model dumps under `RF_model_objects/`, totalling many GB) that
predates this repository and is not published here — those intermediate
artifacts were the private working files from the initial exploratory
analysis, not the final published models. This is deliberate: publishing
every historical intermediate pickle is impractical and unnecessary once the
final models are already published in `models/`.

**The fully reproducible entry points in this repository are:**
- `NAR_main_version_30.8.26.ipynb` — the complete original analysis notebook.
- `models/` — the final trained models, each with a documented preprocessing pipeline.
- `examples/predict_new_transcript.ipynb` — runs end-to-end from this repository alone, no external dependencies.

Running these `scripts/*.py` files as-is on another machine will fail on the
hardcoded path to that external directory. If you want to see *how* a
specific model was produced, read the script; if you want to *use* the
models, use `examples/predict_new_transcript.ipynb` instead.

## What each script does

| Script | Purpose |
|---|---|
| `01_retrain_missing_models.py` | Retrains Model 1 (regulatory-features-only classifier) from scratch, matching the original notebook's methodology. |
| `02_retrain_xgb_regressor.py` | Retrains Model 2 (XGBoost TE regressor). Documents the exact recipe (split 8, full 5'+3'UTR features, asymmetric outlier trim) recovered by tracing the original interactive notebook, and the resulting reproduction gap vs. the manuscript's reported r=0.69. |
| `02b_split1_check.py` | Early diagnostic: checked whether split 1 also reproduces Model 2's reported performance (it does not, split 8 is specific). |
| `02c_seed_sweep.py` | Confirms Model 2's XGBoost result is fully deterministic given fixed data (30-seed sweep, identical result every time) - rules out random initialization as the source of the residual reproduction gap. |
| `03_cv_models3and4.py` | Computes genuine held-out cross-validation performance (accuracy, ROC-AUC, F1) for Models 3 and 4 across all 10 pre-specified data splits, and retrains Model 4 fresh (its original pickle requires an incompatible scikit-learn version). |
| `04_cv_model5_fig6combo.py` | Attempted the same held-out CV reconstruction for Model 5; documents why it is not reliable for this model (extreme feature sparsity in the 3-way combination features) and why Model 5's performance is instead reported as originally described in the manuscript. |
| `05_build_model3_bundle.py` | Packages Model 3 as a self-contained, directly loadable bundle (model + scaler + imputer + feature list). |
| `06_rebuild_model4_split0.py` | Packages Model 4 the same way, with the lightweight `SimpleImputer` substitution. |
| `07_build_model_metadata.py` | Generates `models/model_metadata.json` programmatically from the actual saved bundles and CV tables (not hand-typed) - the single source of truth for every number reported about the models. |
| `08_build_examples_notebook.py` / `09_execute_examples_notebook.py` | Build and execute `examples/predict_new_transcript.ipynb`. |
