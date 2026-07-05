"""
utils.py — Shared module for TCC Weathering ML Pipeline
=========================================================
Schema v2.2 | Redesigned 07/Apr/2026

Imported by NB00–NB09. No hardcoded results — all results come from the database.

Sections:
    1. Paths and constants
    2. Database connection
    3. Data loading (ML dataset, properties, kinetics)
    4. Diagnostic ratios (safe division)
    5. ML pipeline (LOOO, save results)
    6. Clustering (GMM)
    7. Visualization (style, colors, abbreviations)
"""

import sqlite3
import json
import hashlib
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
import optuna


# =============================================================
# 1. PATHS AND CONSTANTS
#
# Single source of truth for all project paths. Every notebook imports
# these instead of defining its own. PROJECT_ROOT is computed from
# __file__ with a fallback for alternate directory layouts.
# SEED=42 ensures reproducibility across LOOO, Optuna, and GMM.
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # utils.py expected in notebooks/
if not (PROJECT_ROOT / 'data').exists():
    # Fallback: utils.py might be at project root level
    PROJECT_ROOT = Path(__file__).resolve().parent
    if not (PROJECT_ROOT / 'data').exists():
        raise RuntimeError(
            f"Cannot locate project root (expected 'data/' directory). "
            f"utils.py should be in PROJECT_ROOT/notebooks/. "
            f"Current location: {Path(__file__).resolve()}"
        )
DB_PATH      = PROJECT_ROOT / 'data' / 'processed' / 'weathering.db'
FIG_ROOT     = PROJECT_ROOT / 'figures'  # each NB defines FIG_DIR = FIG_ROOT / 'nbXX'
MODEL_DIR    = PROJECT_ROOT / 'models' / 'looo_models'
MAPPING_PATH = PROJECT_ROOT / 'compound_name_mapping.md'

SEED = 42

STAGE_MAP = {'W0': 0, 'W1': 1, 'W2': 2, 'W3': 3}
STAGES_ANALYSIS = ['W0', 'W1', 'W2', 'W3']

# Components for sum-based diagnostic ratios (LMW/HMW)
# These are DEFINITIONS (how the ratio is computed), not results.
SUM_RATIO_COMPONENTS = {
    'LMW_HMW_alk': {
        'numerator': ['n-C9', 'n-C10', 'n-C11', 'n-C12', 'n-C13',
                       'n-C14', 'n-C15', 'n-C16', 'n-C17'],
        'denominator': ['n-C25', 'n-C26', 'n-C27', 'n-C28', 'n-C29',
                         'n-C30', 'n-C31', 'n-C32', 'n-C33', 'n-C34', 'n-C35'],
    },
    'LMW_HMW_PAH_5ring': {
        'numerator': ['C0-Naphthalene', 'C1-Naphthalene', 'C2-Naphthalene',
                       'C3-Naphthalene', 'C4-Naphthalene'],
        'denominator': ['C0-Chrysene', 'C1-Chrysene', 'C2-Chrysene', 'C3-Chrysene'],
    },
}


# =============================================================
# 2. DATABASE CONNECTION
#
# Context manager that enforces FK constraints, auto-commits on
# success, and rolls back on exception. Used by NB01-NB09.
# NB00 uses sqlite3.connect() directly (executescript compatibility).
# =============================================================

@contextmanager
def get_conn(db_path=None):
    """
    Context manager for SQLite connection with FK enforcement.
    Auto-commits on success, rollback on exception, always closes.

    Usage:
        with get_conn() as conn:
            conn.execute("SELECT ...")

    Note: For executescript() calls (e.g., NB00 schema creation), use
    sqlite3.connect() directly — executescript() issues implicit commits
    that are incompatible with this context manager's rollback logic.
    """
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Legacy alias
managed_conn = get_conn


# =============================================================
# 2b. DEFENSE-IN-DEPTH GUARDS
#
# Consolidated assertion helpers that consumer NBs call at startup
# to catch silent regressions of upstream patches before producing
# misleading downstream artifacts. See AUDIT_NOTEBOOKS §"Cascata
# F-NB01-C1 → consolidação" for the cascade these guards protect.
# =============================================================

# Canonical sterane compound names (post-CHG-0005/0006). The ß spelling
# is the SSOT in `compounds.compound_name`. Pre-CHG-0005 NBs hardcoded
# the ASCII transliteration `ss(H)`, which silently broke every string-
# match downstream (cascade F-NB01-C1 → F-NB02-C1 → F-NB03c3-C1 →
# F-NB03c4-C1 → F-NB03f-C1).
STERANE_CANONICAL_COMPOUNDS = (
    '14ß(H),17ß(H)-20-Cholestane (C27aßß)',
    '20-Methyl-14ß(H),17ß(H)-Cholestane (C28aßß)',
    '20-Ethyl-14ß(H),17ß(H)-Cholestane (C29aßß)',
)

# Snapshot of the ratios that CHG-0007 reclassified from NOT_AVAILABLE
# to category='canonical'. Used as a SUBSET assertion (these four MUST
# still be present and canonical), independent of any future additions.
# Pre-CHG-0007 these were silently absent because the string-match for
# compound lookup failed on the ß→ss encoding mismatch.
STERANE_CANONICAL_RATIOS = (
    'C27est_C29est',
    'C28est_C29est',
    'C27est_H30',
    'C29est_H30',
)

# Reference biomarker compounds for the pattern-based check (layer 3).
# A ratio whose numerator AND denominator are both in this pool is
# structurally a biomarker × biomarker identity ratio and therefore
# expected to be category='canonical'. Currently the 3 sterane canonical
# compounds plus Hopane (H30); extend if future CHGs introduce additional
# biomarker reference points (e.g. H29). Note: the existing hopane × hopane
# canonical ratios (H29_H30, H31S_H31R, etc.) are intentionally NOT
# covered by this pool — they pre-date CHG-0007 and are out of scope.
_STERANE_BIOMARKER_REFERENCE_POOL = (
    *STERANE_CANONICAL_COMPOUNDS,
    'Hopane (H30)',
)

# ratio_definitions integrity triggers installed by FIX-NB00-1 / CHG-0006.
# Note: these triggers are TABLE-WIDE — they validate every INSERT into
# ratio_definitions, not just sterane rows. They are checked here for
# defense-in-depth convenience (the NBs that consume steranes are also
# the NBs that depend on ratio_definitions integrity), not because they
# are sterane-specific.
_RATIO_DEFS_REQUIRED_TRIGGERS = frozenset({
    'ratio_defs_validate_numerator',
    'ratio_defs_validate_denominator',
})


def assert_sterane_canonical_in_db(conn=None):
    """
    Defense-in-depth guard for the three layers of state that the
    F-NB01-C1 → F-NB02-C1 → F-NB03c3/c4/f-C1 cascade left fragile.

    Despite the name, this function checks three layers of state. Two are
    sterane-specific; the third (validation triggers) is table-wide and is
    folded in here because the same NBs that depend on sterane state also
    depend on ratio_definitions integrity, and a single fail-fast call is
    cleaner than two separate ones.

    LAYER 1 — Compound name encoding (sterane-specific)
        F-NB01-C1 / CHG-0005. The 3 canonical sterane compound names in
        STERANE_CANONICAL_COMPOUNDS must be present in `compounds` with
        the ß spelling (not the ASCII `ss(H)` transliteration that the
        2025 ECCC ESTS CSV produced under cp1252 decoding).

    LAYER 2 — ratio_definitions integrity triggers (TABLE-WIDE, not sterane-specific)
        FIX-NB00-1 / CHG-0006. The two triggers in _RATIO_DEFS_REQUIRED_TRIGGERS
        must be installed on `ratio_definitions`. They abort any INSERT whose
        numerator/denominator does not match a `compounds.compound_name` row,
        preventing the silent "defined-but-empty" ratio failure mode that
        F-NB01-C1 originally produced.

    LAYER 3 — Sterane ratio classification
        CHG-0007. Two complementary checks:
        (3a) SNAPSHOT — the 4 names in STERANE_CANONICAL_RATIOS must still
             have category='canonical'. Catches direct undo of CHG-0007.
        (3b) PATTERN — every ratio in `ratio_definitions` whose numerator
             AND denominator are both in _STERANE_BIOMARKER_REFERENCE_POOL
             must have category='canonical'. Catches future drift: if a
             new sterane-biomarker ratio is added with the wrong category,
             this check fires even though the snapshot list does not list
             it. The snapshot is the floor; the pattern is the ceiling.

    Replaces the manual guards F-NB02-C2 (NB02 cell 4) and F-NB03c3-C2
    (NB03c3 cell 3). Should be called at the top of any NB that consumes
    sterane-related state (compounds, ratios, or SHAP groups).

    Parameters
    ----------
    conn : sqlite3.Connection or None
        If None, opens a transient connection via get_conn(). Pass an
        existing connection if the NB already has one open and wants the
        guard to read uncommitted state (e.g. negative tests using a
        savepoint to mutate-then-rollback).

    Raises
    ------
    AssertionError
        With a diagnostic message naming the responsible CHG/patch and
        the layer of the cascade that regressed.
    """
    def _check(c):
        # ── Layer 1 — sterane compound names (encoding) ──
        placeholders = ','.join(['?'] * len(STERANE_CANONICAL_COMPOUNDS))
        rows = c.execute(
            f"SELECT compound_name FROM compounds WHERE compound_name IN ({placeholders})",
            STERANE_CANONICAL_COMPOUNDS,
        ).fetchall()
        present = {r[0] for r in rows}
        missing = set(STERANE_CANONICAL_COMPOUNDS) - present
        assert not missing, (
            f"[Layer 1] Sterane compounds missing with canonical ß spelling: "
            f"{sorted(missing)}. Likely regression of F-NB01-C1 / CHG-0005 "
            "(ß→ss encoding in NB01). Re-run NB01; FIX-NB01-1 reconciliation "
            "guard should have caught this."
        )

        # ── Layer 2 — ratio_definitions validation triggers (TABLE-WIDE) ──
        triggers = {
            r[0] for r in c.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='trigger' AND tbl_name='ratio_definitions'"
            ).fetchall()
        }
        missing_triggers = _RATIO_DEFS_REQUIRED_TRIGGERS - triggers
        assert not missing_triggers, (
            f"[Layer 2] ratio_definitions validation triggers missing: "
            f"{sorted(missing_triggers)}. Expected from NB00 FIX-NB00-1 "
            "(CHG-0006). Re-run NB00."
        )

        # ── Layer 3a — snapshot: CHG-0007 ratios must still be canonical ──
        placeholders_r = ','.join(['?'] * len(STERANE_CANONICAL_RATIOS))
        rows_r = c.execute(
            f"SELECT ratio_name, category FROM ratio_definitions "
            f"WHERE ratio_name IN ({placeholders_r})",
            STERANE_CANONICAL_RATIOS,
        ).fetchall()
        cats = dict(rows_r)
        non_canonical_snapshot = {
            rn: cats.get(rn, '<MISSING>')
            for rn in STERANE_CANONICAL_RATIOS
            if cats.get(rn) != 'canonical'
        }
        assert not non_canonical_snapshot, (
            f"[Layer 3a snapshot] CHG-0007 sterane ratios not classified as "
            f"'canonical': {non_canonical_snapshot}. Likely regression of "
            "CHG-0007 (NB02 sterane reclassification). Re-run NB02."
        )

        # ── Layer 3b — pattern: any biomarker × biomarker ratio must be canonical ──
        pool = _STERANE_BIOMARKER_REFERENCE_POOL
        ph_pool = ','.join(['?'] * len(pool))
        rows_pat = c.execute(
            f"SELECT ratio_name, category FROM ratio_definitions "
            f"WHERE numerator IN ({ph_pool}) AND denominator IN ({ph_pool})",
            (*pool, *pool),
        ).fetchall()
        non_canonical_pattern = [
            (rn, cat) for rn, cat in rows_pat if cat != 'canonical'
        ]
        assert not non_canonical_pattern, (
            f"[Layer 3b pattern] biomarker × biomarker ratios with wrong category: "
            f"{non_canonical_pattern}. A ratio whose numerator and denominator are "
            "both in the sterane biomarker reference pool must be category='canonical'. "
            "Likely a new sterane ratio added without canonical classification — "
            "investigate the most recent CHG that touched ratio_definitions."
        )

        return (
            len(present),
            len(triggers & _RATIO_DEFS_REQUIRED_TRIGGERS),
            len(STERANE_CANONICAL_RATIOS),
            len(rows_pat),
        )

    if conn is None:
        with get_conn() as _c:
            n_compounds, n_triggers, n_snapshot, n_pattern = _check(_c)
    else:
        n_compounds, n_triggers, n_snapshot, n_pattern = _check(conn)

    print(
        f"✓ assert_sterane_canonical_in_db: "
        f"L1={n_compounds} steranes (ß), "
        f"L2={n_triggers} triggers, "
        f"L3a={n_snapshot} snapshot ratios, "
        f"L3b={n_pattern} biomarker×biomarker ratios (all canonical)"
    )


# =============================================================
# 3. DATA LOADING
#
# Functions to load data from the database into pandas DataFrames.
# load_ml_dataset() is the primary interface for NB03-NB07 — it
# pivots measurements+ratios into the (X, y, meta) format that
# the ML pipeline expects. The new load_properties/kinetics/pan_evap
# functions access the physical/behavioral data added in schema v2.2.
# =============================================================

def load_ml_dataset(conn, include_compounds=True, include_ratios=True,
                    only_crude=False, exclude_features=None):
    """
    Load ML-ready dataset from database.

    Returns (X, y, meta) where:
        X    : DataFrame (n_samples × n_features), may contain NaN
        y    : Series of integer weathering stages (0-3)
        meta : DataFrame with oil_id, oil_name, oil_type, stage_code

    Parameters
    ----------
    conn : sqlite3.Connection
    include_compounds : bool — include individual compound features
    include_ratios : bool — include diagnostic ratio features
    only_crude : bool — filter to oil_type='crude' only
    exclude_features : list or None — feature names to drop (e.g., from r>0.95 filter)
    """
    # Load compounds (pivoted)
    dfs = []

    if include_compounds:
        df_c = pd.read_sql("""
            SELECT o.oil_id, o.oil_name, o.oil_type, m.stage_code,
                   c.compound_name, m.value_imputed
            FROM measurements m
            JOIN oils o ON m.oil_id = o.oil_id
            JOIN compounds c ON m.compound_id = c.compound_id
            WHERE o.include_in_analysis = 1
              AND c.excluded = 0
              AND m.stage_code IN ('W0','W1','W2','W3')
        """, conn)
        if only_crude:
            df_c = df_c[df_c['oil_type'] == 'crude']
        pivot_c = df_c.pivot_table(
            index=['oil_id', 'oil_name', 'oil_type', 'stage_code'],
            columns='compound_name',
            values='value_imputed'
        )
        if pivot_c.empty:
            warnings.warn('measurements table returned no data. Has NB01 been run?')
        dfs.append(pivot_c)

    if include_ratios:
        df_r = pd.read_sql("""
            SELECT o.oil_id, o.oil_name, o.oil_type, dr.stage_code,
                   dr.ratio_name, dr.value
            FROM diagnostic_ratios dr
            JOIN oils o ON dr.oil_id = o.oil_id
            WHERE o.include_in_analysis = 1
              AND dr.is_valid = 1
              AND dr.stage_code IN ('W0','W1','W2','W3')
        """, conn)
        if only_crude:
            df_r = df_r[df_r['oil_type'] == 'crude']
        pivot_r = df_r.pivot_table(
            index=['oil_id', 'oil_name', 'oil_type', 'stage_code'],
            columns='ratio_name',
            values='value'
        )
        if pivot_r.empty:
            warnings.warn('diagnostic_ratios table returned no data. Has NB02 been run?')
        dfs.append(pivot_r)

    if not dfs:
        raise ValueError("At least one of include_compounds or include_ratios must be True")

    combined = pd.concat(dfs, axis=1)
    combined = combined.reset_index()

    # Build meta, y, X
    meta = combined[['oil_id', 'oil_name', 'oil_type', 'stage_code']].copy()
    y = meta['stage_code'].map(STAGE_MAP).astype(int)
    X = combined.drop(columns=['oil_id', 'oil_name', 'oil_type', 'stage_code'])

    # Exclude features if requested
    if exclude_features:
        to_drop = [f for f in exclude_features if f in X.columns]
        X = X.drop(columns=to_drop)

    return X, y, meta


def load_properties(conn, property_names, stage='W0', included_only=True):
    """
    Load physical/chemical properties from sample_properties table.

    Returns DataFrame with oil_id, oil_name, and one column per property.
    """
    placeholders = ','.join(['?'] * len(property_names))
    filter_clause = "AND o.include_in_analysis = 1" if included_only else ""

    df = pd.read_sql(f"""
        SELECT o.oil_id, o.oil_name, sp.property_name, sp.value
        FROM sample_properties sp
        JOIN oils o ON sp.oil_id = o.oil_id
        WHERE sp.property_name IN ({placeholders})
          AND sp.stage_code = ?
          {filter_clause}
    """, conn, params=list(property_names) + [stage])

    if df.empty:
        return df

    pivot = df.pivot_table(
        index=['oil_id', 'oil_name'],
        columns='property_name',
        values='value'
    ).reset_index()
    pivot.columns.name = None
    return pivot


def load_pan_evaporation(conn, oil_id=None, included_only=True):
    """
    Load pan evaporation time series.

    Returns DataFrame with oil_id, oil_name, time_hours, mass_loss_pct.
    If oil_id specified, returns for that oil only.
    """
    filter_clause = "AND o.include_in_analysis = 1" if included_only else ""
    oil_clause = f"AND p.oil_id = {int(oil_id)}" if oil_id else ""

    return pd.read_sql(f"""
        SELECT p.oil_id, o.oil_name, p.time_hours, p.mass_loss_pct
        FROM pan_evaporation p
        JOIN oils o ON p.oil_id = o.oil_id
        WHERE 1=1 {oil_clause} {filter_clause}
        ORDER BY p.oil_id, p.time_hours
    """, conn)


def load_kinetics(conn, included_only=True):
    """
    Load W→time mapping from oil_weathering_kinetics.

    Returns DataFrame with oil_id, oil_name, stage_code, time_hours, mass_loss_pct.
    """
    filter_clause = "AND o.include_in_analysis = 1" if included_only else ""
    return pd.read_sql(f"""
        SELECT k.oil_id, o.oil_name, k.stage_code, k.time_hours, k.mass_loss_pct
        FROM oil_weathering_kinetics k
        JOIN oils o ON k.oil_id = o.oil_id
        WHERE 1=1 {filter_clause}
        ORDER BY k.oil_id, k.stage_code
    """, conn)


# =============================================================
# 4. DIAGNOSTIC RATIOS
#
# Protected division for computing diagnostic ratios (NB02).
# safe_ratio handles None/NaN/zero denominators gracefully.
# The vectorized versions operate on pandas Series for efficiency.
# ECCC data is semi-quantitative (GC-MS peak areas relative to
# internal standards); ratios cancel the response factor.
# =============================================================

def safe_ratio(numerator, denominator):
    """
    Compute ratio with protection against zero/None/NaN denominators.

    Returns (value, is_valid) where:
        value    : float or None
        is_valid : 1 if computed normally, 0 if denominator was invalid
    """
    if numerator is None or denominator is None:
        return (None, 0)
    try:
        num = float(numerator)
        den = float(denominator)
    except (ValueError, TypeError):
        return (None, 0)
    if np.isnan(num) or np.isnan(den) or den <= 0:
        return (None, 0)
    return (num / den, 1)


def safe_ratio_vec(numerator_series, denominator_series):
    """
    Vectorized safe ratio for pandas Series.

    Returns (values, is_valid) as two Series.
    """
    num = pd.to_numeric(numerator_series, errors='coerce')
    den = pd.to_numeric(denominator_series, errors='coerce')
    valid = (den > 0) & den.notna() & num.notna()
    values = pd.Series(np.where(valid, num / den, np.nan), index=num.index)
    is_valid = valid.astype(int)
    return values, is_valid


def safe_sum_ratio_vec(df, numerator_compounds, denominator_compounds):
    """
    Compute ratio of sums: sum(numerators) / sum(denominators).

    Parameters
    ----------
    df : DataFrame with compound columns
    numerator_compounds : list of column names
    denominator_compounds : list of column names

    Returns (values, is_valid) as two Series.
    """
    num_cols = [c for c in numerator_compounds if c in df.columns]
    den_cols = [c for c in denominator_compounds if c in df.columns]
    num_sum = df[num_cols].sum(axis=1, min_count=1)
    den_sum = df[den_cols].sum(axis=1, min_count=1)
    return safe_ratio_vec(num_sum, den_sum)


# =============================================================
# 5. ML PIPELINE — LEAVE-ONE-OIL-OUT CROSS-VALIDATION
#
# The core ML engine. New `run_looo` (28/abr/2026 redesign) is
# preprocessing-agnostic — caller's `model_factory` returns a sklearn-
# compatible estimator (typically a Pipeline encoding scaling/imputation/
# log/PCA per-fold; XGBoost native NaN means no Imputer needed). LOOO
# fold = held-out oil; default drop_w0_missing_oils=True per D1
# (D-W0-CRUDE-42). Persists 3 tables: looo_predictions, looo_metrics,
# looo_model_artifacts (NB04 owns; SQL is authoritative, returned dict
# is this-run-only).
#
# Legacy run_looo / save_looo_results below the new function are
# DEPRECATED (replaced by NB04 redesign 28/abr). Delete after C1 + C8
# produce MAE finite + SHAP-ready predictions persisted end-to-end.
# =============================================================

import hashlib
import pickle
import uuid
from typing import Callable, List, Literal, Optional, Tuple


# =====================================================================
# Pipeline-as-config: feature subsetting helpers (Spec A.2, Sessão P)
# =====================================================================


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Transformer selecting a fixed subset of columns from a DataFrame.

    Used by NB04 model factories to formalize feature subsetting as a
    sklearn Pipeline stage (Pipeline-as-config pattern, D2(b) extension
    formalized in Sessão P).

    Parameters
    ----------
    columns : list of str
        Column names to select. Order is preserved on transform output.

    Notes
    -----
    Validates at transform-time that all requested columns exist in X.
    Raises KeyError if any are missing — silent drops from a feature
    subset configuration would mask data drift.
    """

    def __init__(self, columns):
        self.columns = list(columns)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"ColumnSelector expects DataFrame, got {type(X).__name__}"
            )
        missing = set(self.columns) - set(X.columns)
        if missing:
            sample = sorted(missing)[:5]
            raise KeyError(
                f"ColumnSelector: {len(missing)} columns missing from X: "
                f"{sample}{'...' if len(missing) > 5 else ''}"
            )
        return X[self.columns].copy()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns)


def get_feature_subset_columns(
    conn,
    feature_set_kind: str,
    scope: str,
) -> list:
    """Resolve column names for a model_configs.feature_set value.

    Maps the manifest's feature_set vocabulary (Cohort 1: 'all', 'ratios',
    'compounds', 'identity', 'none') to a sorted list of feature_name
    strings from feature_ml_final, scoped by config ('C45CRUDE' or
    'C62ALL').

    PCA configs ('pca' feature_set) are NOT supported by this helper —
    they are deferred to a follow-up spec block per Path Y decision
    (Sessão P).

    The 'identity' kind queries ratio_definitions.ratio_type='identity'
    intersected with feature_ml_final's ratio rows in the requested
    scope. The empirical N may be ≤ 13 (the manifest-declared count)
    if NB03g's filters dropped any identity ratios; the actual count
    falls out of the query and should be logged at config-load time.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to weathering.db.
    feature_set_kind : str
        One of 'all', 'ratios', 'compounds', 'identity', 'none'.
    scope : str
        'C45CRUDE' or 'C62ALL'. Matches feature_ml_final.config.

    Returns
    -------
    list of str
        Feature column names, sorted alphabetically. Empty list for 'none'.

    Raises
    ------
    ValueError
        If feature_set_kind is 'pca' (deferred), unknown, or scope is
        not in {'C45CRUDE', 'C62ALL'}.
    """
    if feature_set_kind == 'none':
        return []

    if feature_set_kind == 'pca':
        raise ValueError(
            "PCA feature_set is deferred (Path Y decision, Sessão P). "
            "Implement via a dedicated follow-up spec after PCA source "
            "data semantics are empirically resolved."
        )

    if scope not in ('C45CRUDE', 'C62ALL'):
        raise ValueError(
            f"Unknown scope: {scope!r}. Expected 'C45CRUDE' or 'C62ALL'."
        )

    if feature_set_kind == 'all':
        sql = (
            "SELECT feature_name FROM feature_ml_final "
            "WHERE config = ? "
            "ORDER BY feature_name"
        )
        return [r[0] for r in conn.execute(sql, (scope,)).fetchall()]

    if feature_set_kind == 'ratios':
        sql = (
            "SELECT feature_name FROM feature_ml_final "
            "WHERE config = ? AND feature_kind = 'ratio' "
            "ORDER BY feature_name"
        )
        return [r[0] for r in conn.execute(sql, (scope,)).fetchall()]

    if feature_set_kind == 'compounds':
        sql = (
            "SELECT feature_name FROM feature_ml_final "
            "WHERE config = ? AND feature_kind = 'compound' "
            "ORDER BY feature_name"
        )
        return [r[0] for r in conn.execute(sql, (scope,)).fetchall()]

    if feature_set_kind == 'identity':
        sql = (
            "SELECT feature_name FROM feature_ml_final "
            "WHERE config = ? "
            "  AND feature_kind = 'ratio' "
            "  AND feature_name IN ("
            "    SELECT ratio_name FROM ratio_definitions "
            "    WHERE ratio_type = 'identity'"
            "  ) "
            "ORDER BY feature_name"
        )
        return [r[0] for r in conn.execute(sql, (scope,)).fetchall()]

    raise ValueError(
        f"Unknown feature_set_kind: {feature_set_kind!r}. "
        f"Expected one of: 'all', 'ratios', 'compounds', 'identity', 'none'."
    )


def run_looo(
    config_name: str,
    *,
    feature_set: Literal['C45CRUDE', 'C62ALL'],
    model_factory: Callable[[], object],
    feature_loader: Optional[Callable[[sqlite3.Connection, bool], Tuple[pd.DataFrame, pd.Series, pd.DataFrame]]] = None,
    crude_only: bool = True,
    drop_w0_missing_oils: bool = True,
    persist: bool = True,
    persist_models: bool = True,
    db_path=None,
    seed: int = SEED,
    verbose: Literal['silent', 'fold', 'detailed'] = 'fold',
) -> dict:
    """
    Leave-One-Oil-Out cross-validation. Preprocessing-agnostic.

    `model_factory` must return a fresh sklearn-compatible estimator per
    fold (typically a Pipeline encoding scaling/imputation/log/PCA
    per-fold via standard sklearn semantics: Pipeline.fit on training
    fold, Pipeline.predict on test fold — leakage-safe automatically).

    `feature_loader` (optional) injects a custom data-loading callable.
    Default `None` uses the canonical `load_ml_dataset(conn, ...)` path
    against `feature_ml_final`-backed compound/ratio data. When provided,
    must accept `(conn, crude_only)` and return `(X, y, meta)` matching
    `load_ml_dataset`'s contract: `X` a DataFrame indexed by sample,
    `y` a Series of weathering stages, `meta` a DataFrame with columns
    'oil_id', 'oil_name', 'oil_type', 'stage_code' (matching
    `load_ml_dataset`'s return shape). Use cases: synthetic test data,
    alternative feature stores, pre-cached X for repeated experiments.

    Validation:
        feature_set='C45CRUDE' requires crude_only=True (incoherent otherwise);
        feature_set='C62ALL' with crude_only=True allowed as sensitivity sibling.

    Persistence (when persist=True):
        looo_predictions     — per-sample predictions + residuals
        looo_metrics         — per-config aggregate metrics (long-form)
        looo_model_artifacts — pickle filesystem refs (when persist_models=True)
        Pickles to: models/looo_models/{config_name}/fold_{oil_id}.pkl

    Returns dict {
        'predictions':        DataFrame per-sample,
        'fold_metrics':       DataFrame per-fold (derived from predictions),
        'aggregate_metrics':  dict (overall + per-stage + per-oil_type),
        'model_artifacts':    list[dict],
        'run_id':             str (UUID12 linking back to SQL rows),
    }

    Note: returned DataFrames reflect this run only. Downstream notebooks
    (NB05+) MUST read from looo_predictions table for canonical state,
    NOT pickle-import this dict. SQL is authoritative; return value is
    convenience for in-session viz cells.
    """
    # ── Validate feature_set × crude_only consistency ──
    if feature_set == 'C45CRUDE' and not crude_only:
        raise ValueError(
            f"Incoherent: feature_set='C45CRUDE' (features curated for crude-only "
            f"correlation filter) requires crude_only=True. Use feature_set='C62ALL' "
            f"for all-oils training."
        )

    db_path = Path(db_path) if db_path else DB_PATH
    run_id = uuid.uuid4().hex[:12]
    model_dir = MODEL_DIR / config_name
    if persist_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    if verbose != 'silent':
        print(f"run_looo[{config_name}] run_id={run_id}")
        print(f"  feature_set={feature_set} crude_only={crude_only} "
              f"drop_w0_missing={drop_w0_missing_oils}")

    # ── Load feature list + data ──
    with get_conn(db_path) as conn:
        feature_names = pd.read_sql(
            "SELECT feature_name FROM feature_ml_final WHERE config = ?",
            conn, params=(feature_set,),
        )['feature_name'].tolist()

        if feature_loader is None:
            X_full, y_full, meta_full = load_ml_dataset(
                conn,
                include_compounds=True,
                include_ratios=True,
                only_crude=crude_only,
            )
        else:
            X_full, y_full, meta_full = feature_loader(conn, crude_only)

    available = [f for f in feature_names if f in X_full.columns]
    missing = sorted(set(feature_names) - set(X_full.columns))
    if missing:
        warnings.warn(
            f"{len(missing)} features in feature_ml_final[{feature_set}] missing "
            f"from pivoted data. First 5: {missing[:5]}. Continuing with "
            f"{len(available)} available."
        )
    X = X_full[available].reset_index(drop=True)
    y = y_full.reset_index(drop=True)
    meta = meta_full.reset_index(drop=True)

    # ── Drop oils without all 4 stages (D1: drop_w0_missing) ──
    if drop_w0_missing_oils:
        stage_counts = meta.groupby('oil_id')['stage_code'].nunique()
        valid_oils = set(stage_counts[stage_counts == 4].index)
        keep_mask = meta['oil_id'].isin(valid_oils)
        n_dropped_oils = int(meta.loc[~keep_mask, 'oil_id'].nunique())
        n_dropped_samples = int((~keep_mask).sum())
        if n_dropped_samples > 0 and verbose != 'silent':
            dropped_ids = sorted(meta.loc[~keep_mask, 'oil_id'].unique().tolist())
            print(f"  drop_w0_missing_oils: removed {n_dropped_oils} oils "
                  f"({n_dropped_samples} samples). Dropped oil_ids: {dropped_ids}")
        X = X[keep_mask].reset_index(drop=True)
        y = y[keep_mask].reset_index(drop=True)
        meta = meta[keep_mask].reset_index(drop=True)

    oil_ids = sorted(meta['oil_id'].unique().tolist())
    n_oils = len(oil_ids)
    n_features = X.shape[1]

    if verbose != 'silent':
        print(f"  n_oils={n_oils} n_features={n_features} n_samples={len(X)}")

    # ── LOOO loop ──
    np.random.seed(seed)
    pred_rows = []
    artifact_rows = []

    for fold_idx, held_out_oil in enumerate(oil_ids):
        test_mask = meta['oil_id'] == held_out_oil
        X_train, y_train = X[~test_mask], y[~test_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        meta_test = meta[test_mask].reset_index(drop=True)

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        for i in range(len(meta_test)):
            yp = float(y_pred[i])
            yt = int(y_test.iloc[i])
            res = yt - yp
            ae = abs(res)
            pred_rows.append({
                'config_name': config_name,
                'oil_id': int(held_out_oil),
                'stage_code': str(meta_test.iloc[i]['stage_code']),
                'y_true': yt,
                'y_pred': yp,
                'residual': res,
                'abs_error': ae,
                'pm1_correct': int(ae <= 1),
                'fold_idx': int(held_out_oil),  # LOOO: fold_idx == held-out oil_id
                'run_id': run_id,
            })

        if persist_models:
            artifact_path = model_dir / f"fold_{held_out_oil}.pkl"
            with open(artifact_path, 'wb') as f:
                pickle.dump(model, f)
            sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            train_oils = [int(o) for o in oil_ids if o != held_out_oil]
            artifact_rows.append({
                'config_name': config_name,
                'fold_idx': int(held_out_oil),
                'held_out_oil': int(held_out_oil),
                'artifact_path': str(artifact_path.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                'n_features': int(n_features),
                'n_train_samples': int(len(X_train)),
                'train_oil_ids_json': json.dumps(train_oils),
                'sha256': sha,
                'run_id': run_id,
            })

        if verbose in ('fold', 'detailed'):
            fold_mae = float(np.abs(y_pred - y_test.values).mean())
            print(f"  fold {fold_idx + 1:>2}/{n_oils} oil_id={held_out_oil:>4} "
                  f"n_test={len(y_test)} MAE={fold_mae:.3f}")

    df_preds = pd.DataFrame(pred_rows)

    # ── Aggregate metrics (long-form) ──
    agg_rows = []
    n_total = int(len(df_preds))
    overall_mae = float(df_preds['abs_error'].mean())
    overall_rmse = float(np.sqrt((df_preds['residual'] ** 2).mean()))
    overall_pm1 = float(df_preds['pm1_correct'].mean())

    for name, val in [('MAE', overall_mae), ('RMSE', overall_rmse),
                       ('pm1_accuracy', overall_pm1)]:
        agg_rows.append({
            'config_name': config_name, 'metric_name': name,
            'metric_scope': 'all', 'value': val,
            'n_samples': n_total, 'run_id': run_id,
        })

    for stage in ['W0', 'W1', 'W2', 'W3']:
        sub = df_preds[df_preds['stage_code'] == stage]
        if len(sub) > 0:
            agg_rows.append({
                'config_name': config_name, 'metric_name': 'MAE',
                'metric_scope': stage, 'value': float(sub['abs_error'].mean()),
                'n_samples': int(len(sub)), 'run_id': run_id,
            })
            agg_rows.append({
                'config_name': config_name, 'metric_name': 'pm1_accuracy',
                'metric_scope': stage, 'value': float(sub['pm1_correct'].mean()),
                'n_samples': int(len(sub)), 'run_id': run_id,
            })

    pred_with_oiltype = df_preds.merge(
        meta[['oil_id', 'oil_type']].drop_duplicates(), on='oil_id', how='left'
    )
    for ot in pred_with_oiltype['oil_type'].dropna().unique():
        sub = pred_with_oiltype[pred_with_oiltype['oil_type'] == ot]
        if len(sub) > 0:
            agg_rows.append({
                'config_name': config_name, 'metric_name': 'MAE',
                'metric_scope': f'oil_type={ot}', 'value': float(sub['abs_error'].mean()),
                'n_samples': int(len(sub)), 'run_id': run_id,
            })

    df_metrics = pd.DataFrame(agg_rows)
    df_artifacts = pd.DataFrame(artifact_rows) if artifact_rows else pd.DataFrame()

    # ── Persist (last-run-wins per D3(b); FK on oils enforced) ──
    if persist:
        with get_conn(db_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM looo_predictions WHERE config_name = ?", (config_name,))
            cur.execute("DELETE FROM looo_metrics WHERE config_name = ?", (config_name,))
            cur.execute("DELETE FROM looo_model_artifacts WHERE config_name = ?", (config_name,))
            df_preds.to_sql('looo_predictions', conn, if_exists='append', index=False)
            df_metrics.to_sql('looo_metrics', conn, if_exists='append', index=False)
            if not df_artifacts.empty:
                df_artifacts.to_sql('looo_model_artifacts', conn, if_exists='append', index=False)

    # ── Per-fold metrics (derived; not persisted, returned only) ──
    fold_metrics_rows = []
    for oid in oil_ids:
        sub = df_preds[df_preds['oil_id'] == oid]
        if len(sub) > 0:
            fold_metrics_rows.append({
                'oil_id': int(oid),
                'mae': float(sub['abs_error'].mean()),
                'rmse': float(np.sqrt((sub['residual'] ** 2).mean())),
                'pm1_acc': float(sub['pm1_correct'].mean()),
                'n_test': int(len(sub)),
            })
    df_fold_metrics = pd.DataFrame(fold_metrics_rows)

    if verbose != 'silent':
        print(f"  → MAE={overall_mae:.3f} RMSE={overall_rmse:.3f} "
              f"pm1={overall_pm1:.1%} n={n_total}")

    return {
        'predictions': df_preds,
        'fold_metrics': df_fold_metrics,
        'aggregate_metrics': {
            'overall_MAE': overall_mae,
            'overall_RMSE': overall_rmse,
            'overall_pm1_accuracy': overall_pm1,
            'n_samples': n_total,
            'n_folds': n_oils,
        },
        'model_artifacts': artifact_rows,
        'run_id': run_id,
    }


def run_looo_optuna(
    config_name: str,
    *,
    feature_set: Literal['C45CRUDE', 'C62ALL'],
    model_factory: Callable[[optuna.trial.Trial], Pipeline],
    feature_loader: Optional[Callable[[sqlite3.Connection, bool], Tuple[pd.DataFrame, pd.Series, pd.DataFrame]]] = None,
    crude_only: bool = True,
    drop_w0_missing_oils: bool = True,
    persist: bool = True,
    persist_models: bool = True,
    db_path=None,
    seed: int = SEED,
    verbose: Literal['silent', 'fold', 'detailed'] = 'fold',
    n_trials: int = 50,
    n_inner_splits: int = 3,
    direction: Literal['minimize', 'maximize'] = 'minimize',
    show_progress_bar: bool = False,
) -> dict:
    """
    Leave-One-Oil-Out CV with Optuna inner-CV hyperparameter tuning.

    Parallel to run_looo() (Q-NEW-C3 (a) full copy ratification, Sessão P).
    Differs in single substantive way: per outer fold, runs an Optuna study
    over the training oils with inner GroupKFold(n_inner_splits) — preventing
    oil-leakage during HPO — then refits on full training fold using best
    hyperparameters and predicts on the held-out oil.

    `model_factory(trial)` is a closure pattern (Q-NEW-C5 β): factory accepts
    an optuna.trial.Trial and returns a Pipeline. Inner-CV scores each trial;
    final refit calls model_factory(study.best_trial) — FrozenTrial supports
    the suggest_* API for deterministic re-construction.

    feature_set, feature_loader, crude_only, drop_w0_missing_oils, persist,
    persist_models, db_path, seed, verbose: identical semantics to run_looo().

    Persistence (when persist=True):
        looo_predictions      — per-sample predictions + residuals
        looo_metrics          — per-config aggregate metrics (long-form)
        looo_model_artifacts  — pickle filesystem refs (when persist_models=True)
        looo_optuna_runs      — sidecar: per-fold Optuna study summary
                                (config_name, fold_idx, held_out_oil, run_id,
                                 n_trials, best_value, best_params_json,
                                 n_inner_splits, direction, seed)

    Returns dict {
        'predictions':        DataFrame per-sample,
        'fold_metrics':       DataFrame per-fold (derived from predictions),
        'aggregate_metrics':  dict (overall + per-stage + per-oil_type),
        'model_artifacts':    list[dict],
        'optuna_runs':        list[dict] (parallel to persisted sidecar),
        'run_id':             str (UUID12 linking back to SQL rows),
    }

    Note: D3(b) last-run-wins semantics extend to looo_optuna_runs sidecar
    (DELETE WHERE config_name=? + INSERT batch). SQL is authoritative.
    """
    # ── Validate feature_set × crude_only consistency ──
    if feature_set == 'C45CRUDE' and not crude_only:
        raise ValueError(
            f"Incoherent: feature_set='C45CRUDE' (features curated for crude-only "
            f"correlation filter) requires crude_only=True. Use feature_set='C62ALL' "
            f"for all-oils training."
        )

    db_path = Path(db_path) if db_path else DB_PATH
    run_id = uuid.uuid4().hex[:12]
    model_dir = MODEL_DIR / config_name
    if persist_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    if verbose != 'silent':
        print(f"run_looo_optuna[{config_name}] run_id={run_id}")
        print(f"  feature_set={feature_set} crude_only={crude_only} "
              f"drop_w0_missing={drop_w0_missing_oils} "
              f"n_trials={n_trials} n_inner_splits={n_inner_splits}")

    # ── Load feature list + data ──
    with get_conn(db_path) as conn:
        feature_names = pd.read_sql(
            "SELECT feature_name FROM feature_ml_final WHERE config = ?",
            conn, params=(feature_set,),
        )['feature_name'].tolist()

        if feature_loader is None:
            X_full, y_full, meta_full = load_ml_dataset(
                conn,
                include_compounds=True,
                include_ratios=True,
                only_crude=crude_only,
            )
        else:
            X_full, y_full, meta_full = feature_loader(conn, crude_only)

    available = [f for f in feature_names if f in X_full.columns]
    missing = sorted(set(feature_names) - set(X_full.columns))
    if missing:
        warnings.warn(
            f"{len(missing)} features in feature_ml_final[{feature_set}] missing "
            f"from pivoted data. First 5: {missing[:5]}. Continuing with "
            f"{len(available)} available."
        )
    X = X_full[available].reset_index(drop=True)
    y = y_full.reset_index(drop=True)
    meta = meta_full.reset_index(drop=True)

    # ── Drop oils without all 4 stages (D1: drop_w0_missing) ──
    if drop_w0_missing_oils:
        stage_counts = meta.groupby('oil_id')['stage_code'].nunique()
        valid_oils = set(stage_counts[stage_counts == 4].index)
        keep_mask = meta['oil_id'].isin(valid_oils)
        n_dropped_oils = int(meta.loc[~keep_mask, 'oil_id'].nunique())
        n_dropped_samples = int((~keep_mask).sum())
        if n_dropped_samples > 0 and verbose != 'silent':
            dropped_ids = sorted(meta.loc[~keep_mask, 'oil_id'].unique().tolist())
            print(f"  drop_w0_missing_oils: removed {n_dropped_oils} oils "
                  f"({n_dropped_samples} samples). Dropped oil_ids: {dropped_ids}")
        X = X[keep_mask].reset_index(drop=True)
        y = y[keep_mask].reset_index(drop=True)
        meta = meta[keep_mask].reset_index(drop=True)

    oil_ids = sorted(meta['oil_id'].unique().tolist())
    n_oils = len(oil_ids)
    n_features = X.shape[1]

    if verbose != 'silent':
        print(f"  n_oils={n_oils} n_features={n_features} n_samples={len(X)}")

    # ── LOOO loop with Optuna inner-CV ──
    np.random.seed(seed)
    pred_rows = []
    artifact_rows = []
    optuna_run_rows: List[dict] = []

    for fold_idx, held_out_oil in enumerate(oil_ids):
        test_mask = meta['oil_id'] == held_out_oil
        X_train, y_train = X[~test_mask], y[~test_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        meta_test = meta[test_mask].reset_index(drop=True)
        meta_train = meta[~test_mask].reset_index(drop=True)

        # Optuna inner-CV HPO on training oils (GroupKFold prevents oil leakage in HPO)
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

        def _objective(trial: optuna.trial.Trial) -> float:
            inner_cv = GroupKFold(n_splits=n_inner_splits)
            train_groups = meta_train['oil_id'].values
            fold_maes = []
            for inner_tr_idx, inner_val_idx in inner_cv.split(X_train, y_train, groups=train_groups):
                X_inner_tr = X_train.iloc[inner_tr_idx]
                y_inner_tr = y_train.iloc[inner_tr_idx]
                X_inner_val = X_train.iloc[inner_val_idx]
                y_inner_val = y_train.iloc[inner_val_idx]
                m = model_factory(trial)
                m.fit(X_inner_tr, y_inner_tr)
                y_pred_inner = m.predict(X_inner_val)
                fold_maes.append(mean_absolute_error(y_inner_val, y_pred_inner))
            return float(np.mean(fold_maes))

        study.optimize(_objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

        # Record Optuna run for sidecar persistence
        optuna_run_rows.append({
            'config_name': config_name,
            'fold_idx': int(held_out_oil),
            'held_out_oil': int(held_out_oil),
            'run_id': run_id,
            'n_trials': n_trials,
            'best_value': float(study.best_value),
            'best_params_json': json.dumps(study.best_params),
            'n_inner_splits': n_inner_splits,
            'direction': direction,
            'seed': seed,
        })

        # Refit on full training fold using best_trial (FrozenTrial supports suggest API)
        model = model_factory(study.best_trial)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        for i in range(len(meta_test)):
            yp = float(y_pred[i])
            yt = int(y_test.iloc[i])
            res = yt - yp
            ae = abs(res)
            pred_rows.append({
                'config_name': config_name,
                'oil_id': int(held_out_oil),
                'stage_code': str(meta_test.iloc[i]['stage_code']),
                'y_true': yt,
                'y_pred': yp,
                'residual': res,
                'abs_error': ae,
                'pm1_correct': int(ae <= 1),
                'fold_idx': int(held_out_oil),  # LOOO: fold_idx == held-out oil_id
                'run_id': run_id,
            })

        if persist_models:
            artifact_path = model_dir / f"fold_{held_out_oil}.pkl"
            with open(artifact_path, 'wb') as f:
                pickle.dump(model, f)
            sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            train_oils = [int(o) for o in oil_ids if o != held_out_oil]
            artifact_rows.append({
                'config_name': config_name,
                'fold_idx': int(held_out_oil),
                'held_out_oil': int(held_out_oil),
                'artifact_path': str(artifact_path.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                'n_features': int(n_features),
                'n_train_samples': int(len(X_train)),
                'train_oil_ids_json': json.dumps(train_oils),
                'sha256': sha,
                'run_id': run_id,
            })

        if verbose in ('fold', 'detailed'):
            fold_mae = float(np.abs(y_pred - y_test.values).mean())
            print(f"  fold {fold_idx + 1:>2}/{n_oils} oil_id={held_out_oil:>4} "
                  f"n_test={len(y_test)} MAE={fold_mae:.3f} "
                  f"best={study.best_value:.3f} n_trials={n_trials}")

    df_preds = pd.DataFrame(pred_rows)

    # ── Aggregate metrics (long-form) ──
    agg_rows = []
    n_total = int(len(df_preds))
    overall_mae = float(df_preds['abs_error'].mean())
    overall_rmse = float(np.sqrt((df_preds['residual'] ** 2).mean()))
    overall_pm1 = float(df_preds['pm1_correct'].mean())

    for name, val in [('MAE', overall_mae), ('RMSE', overall_rmse),
                       ('pm1_accuracy', overall_pm1)]:
        agg_rows.append({
            'config_name': config_name, 'metric_name': name,
            'metric_scope': 'all', 'value': val,
            'n_samples': n_total, 'run_id': run_id,
        })

    for stage in ['W0', 'W1', 'W2', 'W3']:
        sub = df_preds[df_preds['stage_code'] == stage]
        if len(sub) > 0:
            agg_rows.append({
                'config_name': config_name, 'metric_name': 'MAE',
                'metric_scope': stage, 'value': float(sub['abs_error'].mean()),
                'n_samples': int(len(sub)), 'run_id': run_id,
            })
            agg_rows.append({
                'config_name': config_name, 'metric_name': 'pm1_accuracy',
                'metric_scope': stage, 'value': float(sub['pm1_correct'].mean()),
                'n_samples': int(len(sub)), 'run_id': run_id,
            })

    pred_with_oiltype = df_preds.merge(
        meta[['oil_id', 'oil_type']].drop_duplicates(), on='oil_id', how='left'
    )
    for ot in pred_with_oiltype['oil_type'].dropna().unique():
        sub = pred_with_oiltype[pred_with_oiltype['oil_type'] == ot]
        if len(sub) > 0:
            agg_rows.append({
                'config_name': config_name, 'metric_name': 'MAE',
                'metric_scope': f'oil_type={ot}', 'value': float(sub['abs_error'].mean()),
                'n_samples': int(len(sub)), 'run_id': run_id,
            })

    df_metrics = pd.DataFrame(agg_rows)
    df_artifacts = pd.DataFrame(artifact_rows) if artifact_rows else pd.DataFrame()

    # ── Persist (last-run-wins per D3(b); FK on oils enforced) ──
    if persist:
        with get_conn(db_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM looo_predictions WHERE config_name = ?", (config_name,))
            cur.execute("DELETE FROM looo_metrics WHERE config_name = ?", (config_name,))
            cur.execute("DELETE FROM looo_model_artifacts WHERE config_name = ?", (config_name,))
            df_preds.to_sql('looo_predictions', conn, if_exists='append', index=False)
            df_metrics.to_sql('looo_metrics', conn, if_exists='append', index=False)
            if not df_artifacts.empty:
                df_artifacts.to_sql('looo_model_artifacts', conn, if_exists='append', index=False)

            # Sidecar Optuna runs (D3(b) last-run-wins — DELETE by config + INSERT)
            cur.execute("DELETE FROM looo_optuna_runs WHERE config_name = ?", (config_name,))
            if optuna_run_rows:
                cur.executemany(
                    """
                    INSERT INTO looo_optuna_runs (
                        config_name, fold_idx, held_out_oil, run_id,
                        n_trials, best_value, best_params_json,
                        n_inner_splits, direction, seed
                    ) VALUES (
                        :config_name, :fold_idx, :held_out_oil, :run_id,
                        :n_trials, :best_value, :best_params_json,
                        :n_inner_splits, :direction, :seed
                    )
                    """,
                    optuna_run_rows,
                )

    # ── Per-fold metrics (derived; not persisted, returned only) ──
    fold_metrics_rows = []
    for oid in oil_ids:
        sub = df_preds[df_preds['oil_id'] == oid]
        if len(sub) > 0:
            fold_metrics_rows.append({
                'oil_id': int(oid),
                'mae': float(sub['abs_error'].mean()),
                'rmse': float(np.sqrt((sub['residual'] ** 2).mean())),
                'pm1_acc': float(sub['pm1_correct'].mean()),
                'n_test': int(len(sub)),
            })
    df_fold_metrics = pd.DataFrame(fold_metrics_rows)

    if verbose != 'silent':
        print(f"  → MAE={overall_mae:.3f} RMSE={overall_rmse:.3f} "
              f"pm1={overall_pm1:.1%} n={n_total} "
              f"(Optuna: {n_trials} trials × {n_inner_splits}-fold inner CV × {n_oils} folds)")

    return {
        'predictions': df_preds,
        'fold_metrics': df_fold_metrics,
        'aggregate_metrics': {
            'overall_MAE': overall_mae,
            'overall_RMSE': overall_rmse,
            'overall_pm1_accuracy': overall_pm1,
            'n_samples': n_total,
            'n_folds': n_oils,
        },
        'model_artifacts': artifact_rows,
        'optuna_runs': optuna_run_rows,
        'run_id': run_id,
    }


def compute_loo_conformal_intervals(
    config_name: str,
    *,
    alphas: Tuple[float, ...] = (0.05, 0.10),
    db_path=None,
    persist: bool = True,
    verbose: Literal['silent', 'fold', 'detailed'] = 'fold',
) -> pd.DataFrame:
    """Compute LOO conformal prediction intervals retroactively from looo_predictions.

    For each (oil, stage) prediction, constructs CP interval at level (1-alpha)
    using residuals from all OTHER oils as calibration. Symmetric intervals via
    |residual| quantile; finite-sample-corrected quantile level.

    Method = "leave-one-out conformal" (Vovk 2005; Barber 2021 §3); distinct from
    full jackknife+ which requires an N×N prediction matrix (not retroactively
    available from looo_predictions).

    Coverage guarantee: ≥ (1-alpha) under exchangeability of residuals.

    Sessão Q (Spec D v1) — analysis-content first spec post-Cohort-1.
    """
    db_path = Path(db_path) if db_path else DB_PATH

    with get_conn(db_path) as conn:
        df = pd.read_sql(
            "SELECT oil_id, stage_code, y_true, y_pred, residual, run_id "
            "FROM looo_predictions WHERE config_name = ?",
            conn, params=(config_name,)
        )

    if len(df) == 0:
        raise ValueError(f"No looo_predictions rows for config '{config_name}'.")

    run_id = df['run_id'].iloc[0]
    if df['run_id'].nunique() > 1:
        warnings.warn(
            f"Multiple run_ids in looo_predictions for {config_name}; using first ({run_id}). "
            f"D3(b) last-run-wins should ensure single run_id; check for partial persistence."
        )

    interval_rows = []

    for alpha in alphas:
        target_coverage = 1 - alpha

        for _, row in df.iterrows():
            # Calibration: residuals from oils OTHER than this one
            calibration = df[df['oil_id'] != row['oil_id']]
            abs_residuals = calibration['residual'].abs().values
            n_cal = len(abs_residuals)

            if n_cal < 1:
                raise ValueError(f"Empty calibration set for oil_id {row['oil_id']}")

            # Finite-sample-corrected quantile level (Vovk 2005)
            q_level = min(np.ceil((n_cal + 1) * target_coverage) / n_cal, 1.0)
            q = float(np.quantile(abs_residuals, q_level))

            lo = float(row['y_pred']) - q
            hi = float(row['y_pred']) + q
            in_interval = int(lo <= row['y_true'] <= hi)

            interval_rows.append({
                'config_name': config_name,
                'oil_id': int(row['oil_id']),
                'stage_code': str(row['stage_code']),
                'alpha': float(alpha),
                'y_pred': float(row['y_pred']),
                'lo': lo,
                'hi': hi,
                'in_interval': in_interval,
                'interval_width': hi - lo,
                'n_calibration': n_cal,
                'method': 'loo_conformal_symmetric',
                'run_id': run_id,
            })

    df_intervals = pd.DataFrame(interval_rows)

    if persist:
        with get_conn(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM looo_prediction_intervals WHERE config_name = ?",
                (config_name,)
            )
            df_intervals.to_sql(
                'looo_prediction_intervals', conn, if_exists='append', index=False
            )

    if verbose != 'silent':
        for alpha in alphas:
            sub = df_intervals[df_intervals['alpha'] == alpha]
            empirical = sub['in_interval'].mean()
            mean_width = sub['interval_width'].mean()
            print(f"  α={alpha:.2f} target={1-alpha:.0%} empirical={empirical:.1%} "
                  f"mean_width={mean_width:.3f} n={len(sub)}")

    return df_intervals


# =============================================================
# 5b. LEGACY ML PIPELINE (DEPRECATED — delete after C1+C8 validated)
# =============================================================

def run_looo_legacy_v1(X, y, meta, model_fn, config_name, seed=SEED):
    """
    DEPRECATED: replaced by run_looo (NB04 redesign 28/abr/2026).
    Kept as architectural reference during NB04 redesign.
    Delete after C1 + C8 produce MAE finite + SHAP-ready predictions
    persisted end-to-end via the new run_looo.

    Original Leave-One-Oil-Out CV. See git history pre-Sessão B for
    original docstring. Caller passes (X, y, meta) explicitly and a
    callable model_fn returning (model, params_dict).
    """
    oil_ids = meta['oil_id'].unique()
    metrics_list = []
    preds_list = []

    for oil_id in oil_ids:
        mask_test = meta['oil_id'] == oil_id
        X_train, y_train = X[~mask_test], y[~mask_test]
        X_test, y_test = X[mask_test], y[mask_test]
        meta_test = meta[mask_test]

        model, params = model_fn(X_train, y_train)
        y_pred = model.predict(X_test)

        abs_errors = np.abs(y_pred - y_test.values)
        mae = abs_errors.mean()
        rmse = np.sqrt((abs_errors ** 2).mean())
        ss_res = ((y_test.values - y_pred) ** 2).sum()
        ss_tot = ((y_test.values - y_test.values.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        pct_1 = (abs_errors <= 1).mean() * 100

        metrics_list.append({
            'config': config_name,
            'test_oil_id': int(oil_id),
            'mae': float(mae),
            'rmse': float(rmse),
            'r_squared': float(r2),
            'pct_within_1': float(pct_1),
            'n_test_samples': int(len(y_test)),
            'model_params': json.dumps(params) if params else None,
        })

        for i, idx in enumerate(meta_test.index):
            preds_list.append({
                'oil_id': int(meta_test.loc[idx, 'oil_id']),
                'stage_code': meta_test.loc[idx, 'stage_code'],
                'y_true': float(y_test.loc[idx]),
                'y_pred': float(y_pred[i]),
                'abs_error': float(abs_errors[i]),
            })

    df_metrics = pd.DataFrame(metrics_list)
    df_preds = pd.DataFrame(preds_list)
    return df_metrics, df_preds


def save_looo_results_legacy_v1(conn, df_metrics, df_preds, config_name=None):
    """
    DEPRECATED: replaced by run_looo (NB04 redesign 28/abr/2026) which
    persists internally. Delete after C1 + C8 validated end-to-end.

    Original: persist LOOO results to looo_metrics and looo_predictions
    using legacy schema (fold_id PK linkage). New schema uses
    config_name + oil_id + stage_code composite PK in looo_predictions.
    """
    if config_name:
        conn.execute("DELETE FROM looo_predictions WHERE fold_id IN "
                     "(SELECT fold_id FROM looo_metrics WHERE config=?)",
                     (config_name,))
        conn.execute("DELETE FROM looo_metrics WHERE config=?", (config_name,))

    for _, row in df_metrics.iterrows():
        conn.execute(
            """INSERT INTO looo_metrics
               (config, test_oil_id, mae, rmse, r_squared, pct_within_1,
                n_test_samples, model_params)
               VALUES (?,?,?,?,?,?,?,?)""",
            (row['config'], row['test_oil_id'], row['mae'], row['rmse'],
             row['r_squared'], row['pct_within_1'], row['n_test_samples'],
             row.get('model_params'))
        )
        fold_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        fold_preds = df_preds[df_preds['oil_id'] == row['test_oil_id']]
        for _, pred in fold_preds.iterrows():
            conn.execute(
                """INSERT INTO looo_predictions
                   (fold_id, oil_id, stage_code, y_true, y_pred, abs_error)
                   VALUES (?,?,?,?,?,?)""",
                (fold_id, pred['oil_id'], pred['stage_code'],
                 pred['y_true'], pred['y_pred'], pred['abs_error'])
            )


# =============================================================
# 6. CLUSTERING (GMM)
#
# Gaussian Mixture Model selection by BIC for oil clustering (NB08).
# sklearn is imported inside the function to avoid forcing the
# dependency on notebooks that don't use clustering.
# =============================================================

def fit_gmm_with_selection(X, k_range=range(2, 8), n_init=10, seed=SEED):
    """
    Fit Gaussian Mixture Models for multiple k, select by BIC.

    Returns (best_gmm, results_df) where results_df has columns:
        k, bic, aic, labels (as JSON list)
    """
    from sklearn.mixture import GaussianMixture

    results = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, n_init=n_init,
                               random_state=seed, covariance_type='full')
        gmm.fit(X)
        labels = gmm.predict(X)
        results.append({
            'k': k,
            'bic': gmm.bic(X),
            'aic': gmm.aic(X),
            'labels': labels.tolist(),
            'gmm': gmm,
        })

    results_df = pd.DataFrame(results)
    best_idx = results_df['bic'].idxmin()
    best_gmm = results_df.loc[best_idx, 'gmm']
    return best_gmm, results_df


# =============================================================
# 7. VISUALIZATION
#
# Publication-quality figure settings (DPI 300, Arial 10pt) and
# colorblind-safe palettes (Paul Tol bright). Abbreviation functions
# load from compound_name_mapping.md lazily — no hardcoded dicts.
# Each notebook defines FIG_DIR = FIG_ROOT / 'nbXX' for its figures.
# =============================================================

# ── Color palettes (visual conventions, not data) ──
# Paul Tol bright palette — colorblind-safe
COLORS = {
    'blue': '#4477AA', 'cyan': '#66CCEE', 'green': '#228833',
    'yellow': '#CCBB44', 'red': '#EE6677', 'purple': '#AA3377',
    'grey': '#BBBBBB',
}
COLOR_CYCLE = list(COLORS.values())

STAGE_COLORS = {
    'W0': '#4477AA', 'W1': '#66CCEE', 'W2': '#CCBB44', 'W3': '#EE6677',
}

OILTYPE_COLORS = {
    'crude': '#4477AA', 'refined': '#CCBB44', 'synthetic': '#228833',
    'bitumen_blend': '#EE6677', 'biodiesel': '#AA3377',
    'fuel_oil': '#66CCEE', 'bitumen': '#BBBBBB', 'orimulsion': '#000000',
}

CLUSTER_COLORS = {0: '#4477AA', 1: '#CCBB44', 2: '#EE6677', 3: '#228833'}

# ── Figure dimensions (journal standards) ──
FIG_WIDTH_1COL = 3.5   # inches — single column
FIG_WIDTH_2COL = 7.0   # inches — double column
FIG_DPI = 300


def setup_figure_style():
    """Apply publication-quality matplotlib rcParams."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.dpi': FIG_DPI,
        'savefig.dpi': FIG_DPI,
        'font.size': 10,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (FIG_WIDTH_2COL, 4.5),
        'figure.constrained_layout.use': True,
        'savefig.bbox': 'tight',
        'savefig.transparent': False,
    })


# ── Abbreviations (loaded from file, not hardcoded) ──

_COMPOUND_ABBREV = None
_RATIO_ABBREV = None


def _load_mapping_file(path=None):
    """Parse compound_name_mapping.md into two dicts."""
    global _COMPOUND_ABBREV, _RATIO_ABBREV
    p = path or MAPPING_PATH
    if not p.exists():
        warnings.warn(f"Mapping file not found: {p}. Using identity abbreviations.")
        _COMPOUND_ABBREV = {}
        _RATIO_ABBREV = {}
        return

    text = p.read_text(encoding='utf-8')
    _COMPOUND_ABBREV = {}
    _RATIO_ABBREV = {}

    # Parse markdown tables: | db_name | abbreviation | group |
    in_compounds = False
    in_ratios = False
    for line in text.split('\n'):
        if '## Compounds' in line:
            in_compounds = True
            in_ratios = False
            continue
        if '## Diagnostic ratios' in line:
            in_compounds = False
            in_ratios = True
            continue
        if '## ' in line and '##' != line[:3]:
            in_compounds = False
            in_ratios = False
            continue

        if '|' in line and '---' not in line:
            parts = [p.strip() for p in line.split('|')]
            parts = [p for p in parts if p]  # remove empty
            if len(parts) >= 2:
                db_name = parts[0]
                fig_name = parts[1]
                if db_name in ('ECCC name (in database)', 'Database name'):
                    continue  # header row
                if in_compounds:
                    _COMPOUND_ABBREV[db_name] = fig_name
                elif in_ratios:
                    _RATIO_ABBREV[db_name] = fig_name


def abbrev(name):
    """
    Abbreviate a compound or ratio name for figures.
    Loads mapping lazily from compound_name_mapping.md.
    """
    global _COMPOUND_ABBREV, _RATIO_ABBREV
    if _COMPOUND_ABBREV is None:
        _load_mapping_file()
    return _COMPOUND_ABBREV.get(name, _RATIO_ABBREV.get(name, name))


def abbrev_series(names):
    """Abbreviate a list/Series of names."""
    return [abbrev(n) for n in names]


# =============================================================
# 8. DATABASE QUERIES FOR RESULTS
#
# Functions that replace formerly hardcoded constants (S_DIAG_RATIOS,
# CLUSTER_LABELS, etc.) with live queries from the database.
# This ensures notebooks always use current results, not stale values
# from a previous pipeline execution.
# =============================================================

def load_cluster_labels(conn):
    """Load GMM cluster labels from oils table."""
    df = pd.read_sql("""
        SELECT DISTINCT cluster_gmm, oil_type,
               COUNT(*) as n_oils
        FROM oils
        WHERE include_in_analysis = 1 AND cluster_gmm IS NOT NULL
        GROUP BY cluster_gmm, oil_type
        ORDER BY cluster_gmm, n_oils DESC
    """, conn)
    return df


def load_model_summary(conn):
    """Load MAE summary for all configs, aggregated from looo_metrics."""
    return pd.read_sql("""
        SELECT config,
               AVG(mae) as mae_mean,
               COUNT(*) as n_folds
        FROM looo_metrics
        GROUP BY config
        ORDER BY mae_mean
    """, conn)


# =============================================================
# 9. COEFFICIENT OF VARIATION (NB03h CV-comp v2)
#
# Inter-oil CV formula for the CV-comp v2 pipeline (NB03h, Sessao AG
# 20/mai/2026). Per-feature * per-stage * per-cohort dispatch is
# orchestration concern (inline NB03h); this function owns formula +
# unit convention + NaN-handling + n_min + eps guards.
# =============================================================


def compute_inter_oil_cv(values, n_min=3, eps=1e-10):
    """Inter-oil CV = std(ddof=1) / |mean|, decimal-fraction.

    Canonical column ``feature_consistency.cv`` (CV-comp v2 SPEC sec. 3)
    is decimal-fraction; deprecated ``cv_pct`` was percentage. Pure
    formula + unit-convention; caller computes ``reason`` string
    (orchestration concern).

    Parameters
    ----------
    values : array-like
        1D array of feature values across cohort oils at a given stage.
        NaN entries are dropped before computation.
    n_min : int, default 3
        Minimum non-NaN count. Below threshold returns NaN.
    eps : float, default 1e-10
        Floor for ``|mean|``. CV undefined when denominator vanishes.

    Returns
    -------
    cv : float
        Inter-oil CV as decimal fraction, or numpy.nan if guards fail.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < n_min:
        return np.nan
    mean_val = arr.mean()
    if abs(mean_val) < eps:
        return np.nan
    std_val = arr.std(ddof=1)
    return std_val / abs(mean_val)


# =============================================================
# 10. TABLE FINGERPRINTING (NB06 pre-flight)
#
# Canonical SHA-256-based table fingerprint for integrity verification
# and reproducibility binding (NB06 pre-flight, Sessao AI 20/mai/2026).
# Recipe: PRAGMA column order, PK-ordered rows, repr() for REAL,
# str() for INT/TEXT, '\x00' for NULL, '\x1f'/'\x1e' separators,
# SHA-256 truncated to 16 hex chars. Columns with default
# CURRENT_TIMESTAMP auto-excluded (audit/provenance, not data state).
# Matches CHG-0009 V5 registry truncation pattern.
# =============================================================


def canonical_table_fingerprint(conn, table_name, extra_exclude_cols=()):
    """Deterministic SHA-256 fingerprint of a SQLite table's data state.

    Excludes audit/provenance columns (default CURRENT_TIMESTAMP) so the
    hash captures content state, not insertion timing. Rows are ordered
    by composite primary key (in PK-ordinal order) or by all columns if
    no PK exists. Cells serialize as: REAL -> repr(x) (full-precision
    shortest round-trip), INTEGER/TEXT -> str(x), NULL -> '\\x00'
    sentinel. Field separator '\\x1f', row separator '\\x1e' (ASCII
    unit/record separators, non-printable, no collision with data).
    Hash truncated to first 16 hex chars (8 bytes) to match CHG-0009
    V5 registry truncation.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to the database.
    table_name : str
        Table to fingerprint.
    extra_exclude_cols : tuple of str, optional
        Additional column names to exclude beyond auto-excluded
        CURRENT_TIMESTAMP defaults. Default ().

    Returns
    -------
    fingerprint : str
        16-character hex string (first 8 bytes of SHA-256).

    Raises
    ------
    ValueError
        If no columns remain after exclusion.
    """
    pragma = list(conn.execute(f'PRAGMA table_info({table_name})'))
    # PRAGMA columns: (cid, name, type, notnull, dflt_value, pk)
    excluded = set(extra_exclude_cols)
    for cid, name, ctype, notnull, dflt, pk in pragma:
        if dflt == 'CURRENT_TIMESTAMP':
            excluded.add(name)
    cols = [(cid, name, pk) for (cid, name, ctype, notnull, dflt, pk) in pragma
            if name not in excluded]
    if not cols:
        raise ValueError(
            f'{table_name}: no columns remain after exclusion '
            f'(excluded: {sorted(excluded)})'
        )
    col_names = [name for (cid, name, pk) in cols]

    pk_cols = sorted([(pk, name) for (cid, name, pk) in cols if pk > 0])
    if pk_cols:
        order_cols = [name for (pk, name) in pk_cols]
    else:
        order_cols = col_names

    select_sql = (
        f'SELECT {", ".join(col_names)} FROM {table_name} '
        f'ORDER BY {", ".join(order_cols)}'
    )
    cursor = conn.execute(select_sql)

    def serialize_cell(v):
        if v is None:
            return '\x00'
        if isinstance(v, float):
            return repr(v)
        return str(v)

    FIELD_SEP = '\x1f'
    ROW_SEP = '\x1e'
    row_strings = []
    for row in cursor:
        row_strings.append(FIELD_SEP.join(serialize_cell(c) for c in row))
    blob = ROW_SEP.join(row_strings)

    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]



