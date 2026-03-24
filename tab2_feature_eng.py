"""Tab 2: Feature Engineering — with yield-tracking plot after each step."""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    clean_process_features_with_log,
    filter_columns_by_stats,
    plot_yield_tracking,
)


def _find_yield_col(df, keyword="yield"):
    """
    Return the first column whose name contains `keyword` (case-insensitive).
    Falls back to any column containing 'rate' or '%'.
    Returns None if nothing found.
    """
    for col in df.columns:
        if keyword.lower() in col.lower():
            return col
    for col in df.columns:
        if "rate" in col.lower() or "%" in col.lower():
            return col
    return None


def _show_yield_tracking(df, step_label, custom_col=None, batch_col="BatchID"):
    """
    Display yield-tracking chart inside an expander.
    Uses custom_col if provided, otherwise auto-detects.
    """
    col = custom_col or _find_yield_col(df)
    if col is None or col not in df.columns:
        return

    with st.expander(f"📉 Yield 追蹤：{step_label}", expanded=True):
        fig = plot_yield_tracking(df, col, batch_col=batch_col,
                                  title_prefix=f"[{step_label}]")
        if fig:
            st.pyplot(fig)
            plt.close()
        st.caption(f"追蹤欄位：`{col}`")


def render(selected_process_df):
    st.header("特徵工程 & 清理")
    if selected_process_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    # ── Yield column selector ─────────────────────────────────
    numeric_cols = selected_process_df.select_dtypes(include=["number"]).columns.tolist()
    auto_yield   = _find_yield_col(selected_process_df)

    with st.expander("⚙️ Yield 追蹤設定", expanded=False):
        st.markdown("設定要在每個特徵工程步驟後追蹤的欄位（預設自動偵測含 'Yield' 的欄位）。")
        yield_col = st.selectbox(
            "追蹤欄位（Yield / Target）",
            ["(自動偵測)"] + numeric_cols,
            index=0,
            key="fe_yield_col",
        )
    tracked_col = None if yield_col == "(自動偵測)" else yield_col

    # ── Show baseline ─────────────────────────────────────────
    _show_yield_tracking(selected_process_df, "原始資料（基準）", tracked_col)

    st.markdown("""
    **自動執行以下規則：**
    - 🗑️ 過濾含有 `Verification Result` / `No (na)` 關鍵字的欄位
    - ➕ 配對 Max/Min、After/Before、End/Start → 計算差值
    - 🔢 數字編號欄位（如 _1、_2）→ 取平均後合併
    """)

    # ── Step 1: Feature Engineering ──────────────────────────
    if st.button("🔧 執行特徵工程", key="run_fe"):
        with st.spinner("處理中..."):
            clean_df, drop_log = clean_process_features_with_log(
                selected_process_df, id_col="BatchID")
            st.session_state["clean_df"] = clean_df
            n_before = selected_process_df.shape[1]
            n_after  = clean_df.shape[1]
            st.success(f"✅ 完成！{n_before} 欄 → {n_after} 欄（"
                       f"移除/合併 {n_before - n_after + len([r for r in drop_log.to_dict('records') if 'Averaged' in r.get('Reason','')])} 個，"
                       f"新增 {max(0, n_after - n_before)} 個差值欄）")

            c1, c2 = st.columns(2)
            c1.markdown("#### 清理後資料預覽")
            c1.dataframe(clean_df.head(10), width="stretch")
            c2.markdown("#### 刪除/合併記錄")
            c2.dataframe(drop_log, width="stretch", hide_index=True)

    # Show yield tracking after feature engineering
    _cd = st.session_state.get("clean_df")
    if _cd is not None:
        _show_yield_tracking(_cd, "特徵工程後", tracked_col)

        # ── Step 2: Statistical Filtering ────────────────────
        st.markdown("---")
        st.markdown("#### 📉 統計篩選（移除低資訊量欄位）")

        c1, c2, c3 = st.columns(3)
        cv_thresh   = c1.slider("CV 門檻（低於此值剔除）",   0.0, 0.1, 0.01, 0.001, format="%.3f")
        jump_thresh = c2.slider("Jump Ratio 門檻（高於此值剔除）", 0.1, 1.0, 0.5, 0.05)
        acf_thresh  = c3.slider("ACF 門檻（低於此值剔除）",  0.0, 0.5, 0.2, 0.05)

        if st.button("📉 執行統計篩選", key="run_stat_filter"):
            with st.spinner("篩選中..."):
                filtered_df, dropped_info = filter_columns_by_stats(
                    _cd,
                    cv_threshold=cv_thresh,
                    jump_ratio_threshold=jump_thresh,
                    acf_threshold=acf_thresh,
                )
                # Restore BatchID if removed
                if "BatchID" in _cd.columns and "BatchID" not in filtered_df.columns:
                    filtered_df.insert(0, "BatchID", _cd["BatchID"])

                st.session_state["clean_df"] = filtered_df
                st.success(
                    f"✅ 剔除 {len(dropped_info)} 個欄位 → 剩餘 {filtered_df.shape[1]} 欄"
                )

                if dropped_info:
                    st.dataframe(
                        pd.DataFrame(
                            [(k, v) for k, v in dropped_info.items()],
                            columns=["Column", "Reason"],
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                # Show yield tracking after statistical filtering
                _show_yield_tracking(filtered_df, "統計篩選後", tracked_col)

        # Always show current state tracking if we already filtered
        elif "clean_df" in st.session_state and st.session_state["clean_df"] is not _cd:
            _show_yield_tracking(
                st.session_state["clean_df"], "統計篩選後（上次結果）", tracked_col
            )
