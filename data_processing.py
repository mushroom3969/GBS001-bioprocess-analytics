"""
Data processing utilities: loading, splitting, feature engineering, filtering.
"""
import re
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess


def extract_batch_logic(s):
    """Extract last 4-digit YYNN from BatchID for chronological sorting."""
    match = re.search(r"(\d{4})$", str(s))
    return int(match.group(1)) if match else 0


def extract_number(s):
    """Extract first number found in string (general sorting)."""
    match = re.search(r"\d+", str(s))
    return int(match.group()) if match else 0


def load_and_clean_raw(df):
    """
    Rename BatchID column and coerce non-ID columns to numeric.
    IMPROVEMENT: handles missing BatchID gracefully.
    """
    batch_candidates = [c for c in df.columns if "BatchID" in c]
    if batch_candidates:
        df = df.rename(columns={batch_candidates[0]: "BatchID"})
    cols_to_convert = df.columns.difference(["BatchID"])
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")
    return df


def process_step_count(df):
    """Count parameters per process step (columns with ':' separator)."""
    process_cols = [col for col in df.columns if ":" in col]
    steps = [col.split(":")[0] for col in process_cols]
    step_counts = pd.Series(steps).value_counts().reset_index()
    step_counts.columns = ["Process Step", "Parameter Count"]
    return step_counts


def split_process_df(df):
    """Split wide DataFrame into per-process DataFrames by ':' separator."""
    common_cols = list(dict.fromkeys([col for col in df.columns if ":" not in col]))
    process_names = set([col.split(":")[0] for col in df.columns if ":" in col])
    split_dfs = {}
    for process in process_names:
        process_specific_cols = [col for col in df.columns if col.startswith(f"{process}:")]
        current_all_cols = common_cols + process_specific_cols
        process_df = df.loc[:, ~df.columns.duplicated()][current_all_cols].copy()
        process_df.columns = [
            col.split(":")[-1] if ":" in col else col for col in process_df.columns
        ]
        process_df = process_df.dropna(axis=1, how="all")
        split_dfs[process] = process_df
    return split_dfs


def filt_specific_name(df, query):
    """Return columns whose names contain the query string (case-insensitive)."""
    selected_cols = df.columns[df.columns.str.contains(query, case=False)]
    return df[selected_cols]


def smooth_process_data(df, target_cols, id_cols=None, method="loess", frac=0.3, span=10):
    """
    Apply LOESS or EWMA smoothing to specified columns.
    IMPROVEMENT: id_cols uses None default (avoids mutable default argument).
    """
    if id_cols is None:
        id_cols = ["BatchID"]
    existing_ids = [c for c in id_cols if c in df.columns]
    smoothed_df = df[existing_ids].copy()
    x = np.arange(len(df))
    for col in target_cols:
        if col not in df.columns:
            continue
        y = df[col].values
        mask = ~np.isnan(y)
        if method.lower() == "loess":
            if np.sum(mask) > 10:
                res = lowess(y[mask], x[mask], frac=frac)
                res_y = np.full(len(y), np.nan)
                res_y[mask] = res[:, 1]
                smoothed_df[col] = pd.Series(res_y, index=df.index).interpolate(limit_direction="both")
            else:
                smoothed_df[col] = y
        elif method.lower() == "ewma":
            smoothed_df[col] = df[col].ewm(span=span, adjust=False).mean()
    return smoothed_df


def clean_process_features_with_log(df, id_col="BatchID", protected_cols=None):
    """
    Apply three feature engineering rules:
      A) Drop columns with keywords (Verification Result, No (na))
      B) Compute Max-Min / After-Before / End-Start differences
      C) Average numbered duplicates (_1, _2 -> mean)

    BUGFIX: current_cols is refreshed after Rule A so Rule B
    doesn't operate on already-dropped columns.

    Returns (cleaned_df, drop_log_df).
    """
    df = df.loc[:, ~df.columns.duplicated()]
    new_df = df.copy().reset_index(drop=True)
    drop_log = []
    if protected_cols is None:
        protected_cols = []
    whitelist = set([id_col] + protected_cols)

    # Rule A
    target_keywords = ["Verification Result", "No (na)"]
    to_drop_kw = [
        c for c in new_df.columns
        if any(kw in c for kw in target_keywords)
        and not c.strip().lower().endswith("(times)")
        and c not in whitelist
    ]
    for c in to_drop_kw:
        drop_log.append({"Column": c, "Reason": "Keyword Filter"})
    new_df = new_df.drop(columns=to_drop_kw)

    def clean_col_name(name):
        return re.sub(r"\s?\(.*\)$", "", name).strip()

    # Rule B — FIX: use updated column list after Rule A
    pairs = [
        (["Maximum", "Maximun"], ["Minimum", "Minimun"], "Diff_MaxMin"),
        (["After"], ["Before"], "Diff_AfterBefore"),
        (["End"], ["Start"], "Diff_EndStart"),
    ]
    current_cols = new_df.columns.tolist()
    for high_keys, low_keys, suffix in pairs:
        for k_high in high_keys:
            for k_low in low_keys:
                high_cols = [c for c in current_cols if k_high in c]
                for c_h in high_cols:
                    base_h = clean_col_name(c_h).replace(k_high, "")
                    for c_l in current_cols:
                        if k_low in c_l:
                            base_l = clean_col_name(c_l).replace(k_low, "")
                            if base_h == base_l and c_h != c_l:
                                new_col_name = f"{base_h.strip('_')}_{suffix}"
                                if new_col_name not in new_df.columns:
                                    new_df[new_col_name] = new_df[c_h] - new_df[c_l]

    # Rule C
    pattern = r"^(.*)_(\d+)\s?(\(.*\))$"
    group_dict = {}
    for c in new_df.columns:
        if c in whitelist:
            continue
        match = re.match(pattern, c)
        if match:
            base_name, _, unit = match.groups()
            key = f"{base_name.strip('_')} {unit}"
            group_dict.setdefault(key, []).append(c)

    for key, grouped_cols in group_dict.items():
        if len(grouped_cols) > 1:
            for c in grouped_cols:
                drop_log.append({"Column": c, "Reason": f"Averaged into: {key}"})
            new_df[key] = new_df[grouped_cols].mean(axis=1)
            new_df = new_df.drop(columns=grouped_cols)

    return new_df, pd.DataFrame(drop_log)


def filter_columns_by_stats(df, batch_col="BatchID", cv_threshold=0.01,
                             jump_ratio_threshold=0.3, acf_threshold=0.2):
    """
    Remove low-information columns based on CV, Jump Ratio, ACF lag-1.
    BUGFIX: columns with <5 valid samples are now kept instead of silently skipped.
    Returns (filtered_df, dropped_info_dict).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep_cols = []
    dropped_info = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 5:
            keep_cols.append(col)  # too few samples — keep
            continue
        std_val = series.std()
        if std_val == 0 or series.nunique() <= 1:
            dropped_info[col] = "Constant/Zero Variance"
            continue
        mean_val = series.mean()
        cv = (std_val / abs(mean_val)) if mean_val != 0 else float("inf")
        data_range = series.max() - series.min()
        avg_jump = series.diff().abs().mean()
        jump_ratio = (avg_jump / data_range) if data_range != 0 else 0
        acf_1 = series.autocorr(lag=1)
        reasons = []
        if cv < cv_threshold:
            reasons.append(f"Low CV({cv:.4f})")
        if jump_ratio > jump_ratio_threshold:
            reasons.append(f"High Jump({jump_ratio:.2f})")
        if not np.isnan(acf_1) and acf_1 < acf_threshold:
            reasons.append(f"Low ACF({acf_1:.2f})")
        if reasons:
            dropped_info[col] = " & ".join(reasons)
        else:
            keep_cols.append(col)

    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return df[keep_cols + non_numeric], dropped_info


def missing_col_summary(df):
    """Return DataFrame with missing value counts and ratios per column."""
    na_counts = df.isnull().sum()
    na_ratio = (na_counts / len(df)) * 100
    mask = na_counts > 0
    summary = pd.concat(
        [na_counts[mask].sort_values(ascending=False),
         na_ratio[mask].sort_values(ascending=False)],
        axis=1
    )
    summary.columns = ["Missing Count", "Missing Ratio (%)"]
    return summary
