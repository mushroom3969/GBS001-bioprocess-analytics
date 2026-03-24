import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── data_processing ──────────────────────────────────────────────────────────
from utils.data_processing import (
    # Batch ID helpers
    extract_batch_logic,
    extract_number,
    # Loading
    load_and_clean_raw,
    process_step_count,
    split_process_df,
    filt_specific_name,
    # Smoothing
    smooth_process_data,
    # Feature engineering
    clean_process_features_with_log,
    filter_columns_by_stats,
    # Missing value summary
    missing_col_summary,
)

# ── plotting ─────────────────────────────────────────────────────────────────
from utils.plotting import (
    plot_indexed_lineplots,
    plot_clean_lineplots,
    plot_correlation_bar,
    plot_missing_heatmap,
    plot_yield_tracking,
)

# ── ml_analysis ──────────────────────────────────────────────────────────────
from utils.ml_analysis import (
    # Correlation
    compute_correlation,
    # PCA / Hotelling T²
    compute_ht2_thresholds,
    compute_ht2_per_sample,
    compute_total_contribution,
    compute_pc_contribution,
    # Random Forest
    train_rf_and_importance,
    # SHAP helpers
    make_short_feature_map,
    get_shap_base_value,
    restore_shap_yticklabels,
    # PLS-VIP
    compute_pls_vip,
    compute_pls_cv_mse,
    # PubMed + Gemini
    pubmed_search,
    pubmed_fetch_abstracts,
    build_pubmed_queries_with_gemini,
    call_gemini,
)

__all__ = [
    # data_processing
    "extract_batch_logic", "extract_number",
    "load_and_clean_raw", "process_step_count", "split_process_df",
    "filt_specific_name", "smooth_process_data",
    "clean_process_features_with_log", "filter_columns_by_stats",
    "missing_col_summary",
    # plotting
    "plot_indexed_lineplots", "plot_clean_lineplots",
    "plot_correlation_bar", "plot_missing_heatmap", "plot_yield_tracking",
    # ml_analysis
    "compute_correlation",
    "compute_ht2_thresholds", "compute_ht2_per_sample",
    "compute_total_contribution", "compute_pc_contribution",
    "train_rf_and_importance",
    "make_short_feature_map", "get_shap_base_value", "restore_shap_yticklabels",
    "compute_pls_vip", "compute_pls_cv_mse",
    "pubmed_search", "pubmed_fetch_abstracts",
    "build_pubmed_queries_with_gemini", "call_gemini",
]
