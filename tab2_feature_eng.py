"""Tab 2: Feature Engineering"""
import streamlit as st
from utils import clean_process_features_with_log, filter_columns_by_stats


def render(selected_process_df):
    st.header("特徵工程 & 清理")
    if selected_process_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    st.markdown("""
    **自動執行以下規則：**
    - 🗑️ 過濾含有 `Verification Result` / `No (na)` 關鍵字的欄位
    - ➕ 配對 Max/Min、After/Before、End/Start → 計算差值
    - 🔢 數字編號欄位（如 _1、_2）→ 取平均後合併
    """)

    if st.button("🔧 執行特徵工程", key="run_fe"):
        with st.spinner("處理中..."):
            clean_df, drop_log = clean_process_features_with_log(
                selected_process_df, id_col="BatchID")
            st.session_state["clean_df"] = clean_df
            st.success(f"✅ 完成！{selected_process_df.shape[1]} 欄 → {clean_df.shape[1]} 欄")
            c1, c2 = st.columns(2)
            c1.markdown("#### 清理後資料預覽")
            c1.dataframe(clean_df.head(10), width="stretch")
            c2.markdown("#### 刪除/合併記錄")
            c2.dataframe(drop_log, width="stretch", hide_index=True)

    _cd = st.session_state.get("clean_df")
    if _cd is not None:
        st.markdown("---")
        st.markdown("#### 📉 統計篩選（移除低資訊量欄位）")
        c1, c2, c3 = st.columns(3)
        cv_thresh   = c1.slider("CV 門檻", 0.0, 0.1, 0.01, 0.001, format="%.3f")
        jump_thresh = c2.slider("Jump Ratio 門檻", 0.1, 1.0, 0.5, 0.05)
        acf_thresh  = c3.slider("ACF 門檻", 0.0, 0.5, 0.2, 0.05)

        if st.button("📉 執行統計篩選", key="run_stat_filter"):
            with st.spinner("篩選中..."):
                filtered_df, dropped_info = filter_columns_by_stats(
                    _cd, cv_threshold=cv_thresh,
                    jump_ratio_threshold=jump_thresh,
                    acf_threshold=acf_thresh)
                if "BatchID" in _cd.columns and "BatchID" not in filtered_df.columns:
                    filtered_df.insert(0, "BatchID", _cd["BatchID"])
                st.session_state["clean_df"] = filtered_df
                st.success(f"✅ 剔除 {len(dropped_info)} 個欄位 → 剩餘 {filtered_df.shape[1]} 欄")
                if dropped_info:
                    import pandas as pd
                    st.dataframe(
                        pd.DataFrame([(k, v) for k, v in dropped_info.items()],
                                     columns=["Column", "Reason"]),
                        width="stretch", hide_index=True)
