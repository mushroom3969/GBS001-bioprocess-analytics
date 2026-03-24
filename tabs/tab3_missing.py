"""Tab 3: Missing Value Analysis — with yield-tracking after manual removal."""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import matplotlib.pyplot as plt

from utils import missing_col_summary, plot_missing_heatmap, plot_yield_tracking


def _find_yield_col(df, keyword="yield"):
    for col in df.columns:
        if keyword.lower() in col.lower():
            return col
    for col in df.columns:
        if "rate" in col.lower() or "%" in col.lower():
            return col
    return None


def render(work_df):
    st.header("缺失值分析")
    if work_df is None:
        st.info("請先在側欄選擇製程步驟，或執行特徵工程。")
        return

    # ── Yield column selector ─────────────────────────────────
    numeric_cols = work_df.select_dtypes(include=["number"]).columns.tolist()
    auto_yield   = _find_yield_col(work_df)

    with st.expander("⚙️ Yield 追蹤設定", expanded=False):
        yield_col = st.selectbox(
            "追蹤欄位（Yield / Target）",
            ["(自動偵測)"] + numeric_cols,
            index=0,
            key="miss_yield_col",
        )
    tracked_col = None if yield_col == "(自動偵測)" else yield_col

    # ── Show baseline yield tracking ─────────────────────────
    col_to_track = tracked_col or _find_yield_col(work_df)
    if col_to_track and col_to_track in work_df.columns:
        with st.expander("📉 Yield 追蹤（移除前基準）", expanded=True):
            fig = plot_yield_tracking(work_df, col_to_track,
                                      title_prefix="[移除前]")
            if fig:
                st.pyplot(fig); plt.close()

    # ── Missing value summary ─────────────────────────────────
    summary_df = missing_col_summary(work_df)
    if summary_df.empty:
        st.success("🎉 無缺失值！")
    else:
        st.metric("含缺失值的欄位數", len(summary_df))
        st.dataframe(
            summary_df.style.background_gradient(cmap="Reds", subset=["Missing Ratio (%)"]),
            width="stretch",
        )
        st.markdown("#### 缺失值熱圖")
        fig = plot_missing_heatmap(work_df, summary_df.index.tolist())
        st.pyplot(fig); plt.close()

    # ── Manual removal ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🗑️ 手動移除批次 / 欄位")
    if "BatchID" in work_df.columns:
        drop_batches = st.multiselect("選擇要移除的 BatchID", work_df["BatchID"].tolist())
        drop_cols_ui = st.text_input("要移除的欄位名稱（逗號分隔）", "")

        if st.button("🗑️ 執行移除", key="drop_rows"):
            filtered = work_df.copy()
            if drop_batches:
                filtered = filtered[~filtered["BatchID"].isin(drop_batches)]
            if drop_cols_ui.strip():
                cols_to_drop = [
                    c.strip() for c in drop_cols_ui.split(",")
                    if c.strip() in filtered.columns
                ]
                filtered = filtered.drop(columns=cols_to_drop)
            st.session_state["clean_df"] = filtered
            st.success(
                f"✅ 移除後：{filtered.shape[0]} 筆 × {filtered.shape[1]} 欄"
                + (f"（移除 {len(drop_batches)} 批）" if drop_batches else "")
            )
            st.dataframe(filtered.head(), width="stretch")

            # ── Yield tracking after removal ─────────────────
            col_after = tracked_col or _find_yield_col(filtered)
            if col_after and col_after in filtered.columns:
                with st.expander("📉 Yield 追蹤（移除後）", expanded=True):
                    fig = plot_yield_tracking(filtered, col_after,
                                              title_prefix="[移除後]",
                                              color="#e84855")
                    if fig:
                        st.pyplot(fig); plt.close()
                    st.caption(
                        f"移除了 {len(drop_batches)} 個批次，"
                        f"追蹤欄位：`{col_after}`"
                    )
