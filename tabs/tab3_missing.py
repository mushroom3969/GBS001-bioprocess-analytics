"""Tab 3: Missing Value Analysis"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)  # go up from tabs/ to root
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


import streamlit as st
import matplotlib.pyplot as plt
from utils import missing_col_summary, plot_missing_heatmap


def render(work_df):
    st.header("缺失值分析")
    if work_df is None:
        st.info("請先在側欄選擇製程步驟，或執行特徵工程。")
        return

    summary_df = missing_col_summary(work_df)
    if summary_df.empty:
        st.success("🎉 無缺失值！")
    else:
        st.metric("含缺失值的欄位數", len(summary_df))
        st.dataframe(
            summary_df.style.background_gradient(cmap="Reds", subset=["Missing Ratio (%)"]),
            width="stretch")
        st.markdown("#### 缺失值熱圖")
        fig = plot_missing_heatmap(work_df, summary_df.index.tolist())
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### 🗑️ 手動移除批次")
    if "BatchID" in work_df.columns:
        drop_batches  = st.multiselect("選擇要移除的 BatchID", work_df["BatchID"].tolist())
        drop_cols_ui  = st.text_input("要移除的欄位名稱（逗號分隔）", "")
        if st.button("🗑️ 執行移除", key="drop_rows"):
            filtered = work_df.copy()
            if drop_batches:
                filtered = filtered[~filtered["BatchID"].isin(drop_batches)]
            if drop_cols_ui.strip():
                cols_to_drop = [c.strip() for c in drop_cols_ui.split(",")
                                if c.strip() in filtered.columns]
                filtered = filtered.drop(columns=cols_to_drop)
            st.session_state["clean_df"] = filtered
            st.success(f"✅ 移除後：{filtered.shape[0]} 筆 × {filtered.shape[1]} 欄")
            st.dataframe(filtered.head(), width="stretch")
