# Model configuration crosswalk


The internal config codes below are **values stored in the database**
(`model_configs.config`, `looo_predictions.config_name`, …) and **model-folder
names** (`models/looo_models/<code>/`), so they are kept verbatim throughout the
pipeline. This legend maps each code to the name used in the paper (Table S2).

| Code | Paper name | Model / feature scope |
|---|---|---|
| `C1` | XGB-all (mixed) | XGBoost, all features, mixed oil types — cohort `C62ALL` (142 features, 44 oils) |
| `C8` | XGB-all (crude) | XGBoost, all features, crude-only — cohort `C45CRUDE` (127 features, 29 oils) |
| `C2` | XGB-ratios | XGBoost, diagnostic ratios only |
| `C3` | XGB-compounds | XGBoost, compounds only |
| `C2i` | XGB-identity | XGBoost, identity-class ratios only |
| `C6` | RF-all | Random Forest, all features |
| `Ridge` | Ridge | RidgeCV linear baseline |
| `C7` | Dummy (crude) | median-prediction baseline, crude scope |
| `C7_mixed` | Dummy (mixed) | median-prediction baseline, mixed scope |
| `C4`, `C4b`, `C4c` | *(not in paper)* | PCA exploratory configs — defined in `model_configs` but **not run** (no predictions/models on disk); excluded from Table S2 |

**Naming caveats.**
- The numerals in `C62ALL` / `C45CRUDE` are **not** counts — the actual counts are 142/127 features and 44/29 oils. Verify by row count, never by the code numerals.
- `W0`–`W3` are the four weathering stages (used as-is in the paper).
- `Sessão …`, `D-…`, `CHG-…`, `§18` are internal process labels (development sessions / decisions / changelog / pre-registration discipline), retained for provenance.
