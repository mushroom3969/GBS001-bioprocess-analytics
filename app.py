"""
Bioprocess Data Analysis Platform
Entry point — configures page, sidebar, and routes to tab modules.
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import sys
import os

# Make utils and tabs importable
sys.path.insert(0, os.path.dirname(__file__))

from utils import load_and_clean_raw, split_process_df

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bioprocess Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1f6aa5; }
    h2 { color: #2e86ab; border-bottom: 2px solid #e0e0e0; padding-bottom: 4px; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #2e86ab;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
for key in ["raw_df", "dfs_dict", "selected_process_df", "clean_df",
            "pca_model", "fi_rf", "shap_vals", "pls_vip_df",
            "pubmed_results", "lit_response"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dna-helix.png", width=60)
    st.title("🧬 Bioprocess Analytics")
    st.markdown("---")
    st.markdown("### 📁 資料載入")

    uploaded_file = st.file_uploader("上傳 CSV 資料檔", type=["csv"])
    if uploaded_file:
        try:
            import pandas as pd
            raw_df = pd.read_csv(uploaded_file)
            raw_df = load_and_clean_raw(raw_df)
            st.session_state["raw_df"]    = raw_df
            st.session_state["dfs_dict"]  = split_process_df(raw_df)
            st.success(f"✅ 載入成功！{raw_df.shape[0]} 筆 × {raw_df.shape[1]} 欄")
        except Exception as e:
            st.error(f"載入失敗：{e}")

    if st.session_state["dfs_dict"] is not None:
        st.markdown("---")
        st.markdown("### ⚙️ 製程步驟")
        process_list = list(st.session_state["dfs_dict"].keys())
        selected_process = st.selectbox("選擇製程步驟", process_list)
        st.session_state["selected_process"]    = selected_process
        st.session_state["selected_process_df"] = st.session_state["dfs_dict"][selected_process]

    st.markdown("---")
    st.caption("Bioprocess Analytics v2.0")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🧬 Bioprocess Data Analysis Platform")

if st.session_state["raw_df"] is None:
    st.markdown("""
    <div class="info-box">
    👈 請先在左側上傳 <b>CSV 資料檔</b>（raw_data.csv）開始分析。
    </div>
    """, unsafe_allow_html=True)
    st.stop()

raw_df             = st.session_state["raw_df"]
dfs_dict           = st.session_state["dfs_dict"]
selected_process_df= st.session_state.get("selected_process_df")
selected_process   = st.session_state.get("selected_process", "")

# work_df: use clean_df if available, otherwise raw process df
_cd      = st.session_state.get("clean_df")
work_df  = _cd if _cd is not None else selected_process_df

# ── Tabs ──────────────────────────────────────────────────────────────────────
from tabs import (tab0_overview, tab1_trends, tab2_feature_eng,
                  tab3_missing, tab4_correlation, tab5_pca,
                  tab6_feature_importance, tab7_literature)

tabs = st.tabs([
    "📊 資料總覽", "📈 趨勢圖", "🔧 特徵工程",
    "🔍 缺失值分析", "🔗 相關性分析", "🧩 PCA 分析",
    "🌲 特徵重要性", "📚 文獻佐證分析",
])

with tabs[0]: tab0_overview.render(raw_df, dfs_dict, selected_process_df, selected_process)
with tabs[1]: tab1_trends.render(selected_process_df)
with tabs[2]: tab2_feature_eng.render(selected_process_df)
with tabs[3]: tab3_missing.render(work_df)
with tabs[4]: tab4_correlation.render(work_df)
with tabs[5]: tab5_pca.render(work_df)
with tabs[6]: tab6_feature_importance.render(work_df)
with tabs[7]: tab7_literature.render()
