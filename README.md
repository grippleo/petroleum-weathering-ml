# Interpretable machine learning for cross-source estimation of petroleum weathering stage

Deposited analysis code for **Paper B** of the TCC by **Leonardo Gripp Bom Amorim**
(UNICAMP + IEAPM).

> **Author to finalize:** exact paper title, citation, and author/affiliation list.

---

## Overview

This repository contains the analysis pipeline behind the paper. Using the
[ECCC ESTS](#data-provenance) petroleum-products dataset (45 oils, 180 samples across four
evaporative weathering stages **W0–W3**), it:

1. extracts compound concentrations and derives ~130 diagnostic ratios from the raw table
   (notebooks `00`–`02`);
2. performs exploratory geochemical analysis by domain (notebooks `03*`);
3. trains **XGBoost** and baseline models to estimate the weathering stage under
   **leave-one-oil-out (LOOO)** cross-validation (`04`);
4. produces **TreeSHAP** feature attributions and model diagnostics (`05`, `06`).

## Reproducibility model

This is a **"reproduce" deposit**, not a build-from-a-blank-slate pipeline. The ML
notebooks (`04`/`05`/`06`) are *validators* of a frozen run: they load pinned per-fold
models and assert reproduction against the shipped database (`shap_hierarchy` reproduces
at `rtol = 1e-9` under the pinned environment). The repository therefore ships:

- the prebuilt, scrubbed SQLite database `data/processed/weathering.db`;
- the pinned per-fold models `models/looo_models/<config>/*.pkl`;
- the raw source table `data/raw/*.csv`.

Re-running the notebooks reproduces and validates against this shipped state.

## Repository layout

```
.
├── README.md
├── CONFIG_CROSSWALK.md          # internal config codes -> paper names
├── environment.yml              # conda environment (pinned)
├── data/
│   ├── raw/                     # raw ECCC ESTS source table (CSV)
│   └── processed/weathering.db  # prebuilt SQLite database (schema v2.x)
├── models/looo_models/<config>/ # pinned per-fold models (.pkl)
└── notebooks/
    ├── 00_database_setup.ipynb        # schema + initial metadata
    ├── 01_csv_extraction.ipynb        # raw CSV -> database
    ├── 02_diagnostic_ratios.ipynb     # diagnostic ratios
    ├── 03_physical_properties.ipynb   # physical-property derivations
    ├── 03b … 03g                      # EDA by geochemical domain + feature selection
    ├── 04_xgboost_looo.ipynb          # LOOO cross-validation (XGBoost + baselines)
    ├── 05_model_diagnostics.ipynb     # prediction geometry, outliers, peer ranking, Ridge head-to-head
    ├── 06_shap_hierarchy.ipynb        # TreeSHAP hierarchy (reproduction validator)
    └── utils.py                       # shared infrastructure (paths, get_conn, run_looo, …)
```

## Environment

```bash
conda env create -f environment.yml
conda activate tcc-weathering
jupyter lab
```

The `shap` package is deliberately **not** a dependency: TreeSHAP values are computed via
xgboost's native `predict(..., pred_contribs=True)`.

## Running the notebooks

Open the notebooks from **within the `notebooks/` directory** (paths are resolved relative
to the repository root via `utils.py`, and `06` asserts the working directory is
`notebooks/`).

- To **reproduce the ML results**, run `04` → `05` → `06` against the shipped database.
  `06` recomputes `shap_hierarchy` and checks it against the shipped values at `rtol = 1e-9`.
- Notebooks `00`–`03*` **document how the database was built** from the raw CSV. The shipped
  `weathering.db` is authoritative; re-running `00` rebuilds the schema from scratch.

## Configuration codes

The model configuration codes (`C1`, `C8`, `C2`, …) are **values stored in the database and
in model-folder names**, so they are kept verbatim throughout the pipeline. See
[`CONFIG_CROSSWALK.md`](CONFIG_CROSSWALK.md) for the mapping to the names used in the paper
(e.g. `C1` → XGB-all (mixed), `C8` → XGB-all (crude)).

## Internal process labels

Some comments and markdown retain internal provenance labels from the development history —
`Sessão …` (development sessions), `D-…` (decisions), `CHG-…` (database changelog),
`§18` (pre-registration discipline). They document how the analysis evolved and are **not**
required to run the code.

## Data provenance

The source data is the **Environment and Climate Change Canada (ECCC) — Emergencies Science
and Technology Section (ESTS)** petroleum-products database (release 2021-01-22). The raw
table is included under `data/raw/`; the derived, analysis-ready state is in
`data/processed/weathering.db`.

## License

Two licenses apply — see [`LICENSE`](LICENSE) for full text.

- **Code** (notebooks, `utils.py`): **MIT License** © 2026 Leonardo Gripp Bom Amorim.
- **Data** (`data/raw/*.csv` and the derived `data/processed/weathering.db`): derived from
  the ECCC ESTS *Crude Oil and Petroleum Product Database*, used under the
  [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada).
  *Contains information licensed under the Open Government Licence – Canada;* source:
  Environment and Climate Change Canada (ECCC), Emergencies Science and Technology Section.

## Citation

*Author to finalize.* Please cite the associated paper (Leonardo Gripp Bom Amorim,
UNICAMP + IEAPM). A full citation / DOI will be added here on publication.
