"""Tab 1: Trend Plots"""
import streamlit as st
import matplotlib.pyplot as plt
from utils import filt_specific_name, smooth_process_data, plot_indexed_lineplots


def render(selected_process_df):
    st.header("趨勢圖")
    if selected_process_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    col_a, col_b, col_c = st.columns([2, 1, 1])
    keyword      = col_a.text_input("欄位關鍵字篩選（留空 = 全部）", "")
    smooth_method= col_b.selectbox("平滑方法", ["loess", "ewma", "none"])
    cols_per_row = col_c.slider("每列圖數", 1, 5, 3)

    if keyword:
        display_df = filt_specific_name(selected_process_df, keyword)
        if "BatchID" not in display_df.columns and "BatchID" in selected_process_df.columns:
            display_df.insert(0, "BatchID", selected_process_df["BatchID"])
    else:
        display_df = selected_process_df.copy()

    if st.button("🖼️ 繪製趨勢圖", key="plot_trend"):
        if smooth_method != "none":
            num_cols = display_df.select_dtypes(include=["number"]).columns.tolist()
            plot_df = smooth_process_data(display_df, num_cols, method=smooth_method)
            if "BatchID" in display_df.columns:
                plot_df["BatchID"] = display_df["BatchID"].values
        else:
            plot_df = display_df.copy()

        with st.spinner("繪圖中..."):
            fig = plot_indexed_lineplots(plot_df, cols_per_row=cols_per_row)
            if fig:
                st.pyplot(fig)
                plt.close()
