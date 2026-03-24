"""Tab 4: Correlation Analysis"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import matplotlib.pyplot as plt
from utils import compute_correlation, plot_correlation_bar


def render(work_df):
    st.header("相關性分析")
    if work_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    numeric_cols = work_df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        st.warning("無數值型欄位。")
        return

    c1, c2, c3 = st.columns(3)
    target_col = c1.selectbox("目標欄位（Y）", numeric_cols)
    method     = c2.selectbox("相關係數方法", ["pearson", "spearman"])
    top_n      = c3.slider("顯示前 N 個特徵", 5, min(50, len(numeric_cols)), 15)

    if st.button("🔗 計算相關性", key="run_corr"):
        with st.spinner("計算中..."):
            corr_rank = compute_correlation(work_df, target_col, method=method)
            if corr_rank is not None:
                fig = plot_correlation_bar(corr_rank, target_col, top_n, method)
                st.pyplot(fig); plt.close()
                st.markdown("#### 相關係數排行")
                st.dataframe(
                    corr_rank.head(top_n).style.background_gradient(
                        cmap="RdBu_r", subset=["Correlation"], vmin=-1, vmax=1),
                    width="stretch", hide_index=True)
                st.session_state["target_col"]  = target_col
                st.session_state["corr_rank"]   = corr_rank
