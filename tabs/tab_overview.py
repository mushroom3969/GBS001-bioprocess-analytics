"""Tab 0 — 資料總覽"""
import streamlit as st
from utils import process_step_count


def render(raw_df, dfs_dict, selected_process_df, selected_process):
    st.header("資料總覽")

    col1, col2, col3 = st.columns(3)
    col1.metric("總批次數", raw_df.shape[0])
    col2.metric("總欄位數", raw_df.shape[1])
    col3.metric("製程步驟數", len(dfs_dict))

    st.markdown("#### 製程步驟欄位統計")
    st.dataframe(process_step_count(raw_df), width="stretch", hide_index=True)

    st.markdown("#### 原始資料預覽")
    st.dataframe(raw_df.head(10), width="stretch")

    if selected_process_df is not None:
        st.markdown(f"#### 已選製程：`{selected_process}` — 欄位預覽")
        st.dataframe(selected_process_df.head(10), width="stretch")
