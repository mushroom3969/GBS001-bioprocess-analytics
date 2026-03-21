"""
Bioprocess Data Analysis Tool
一個針對生物製藥製程（rhG-CSF）的互動式數據分析平台
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math
import textwrap
import scipy.stats as stats
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore")

from pca import pca
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bioprocess Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def process_step_count(df):
    process_cols = [col for col in df.columns if ":" in col]
    steps = [col.split(":")[0] for col in process_cols]
    step_counts = pd.Series(steps).value_counts().reset_index()
    step_counts.columns = ["Process Step", "Parameter Count"]
    return step_counts


def split_process_df(df):
    common_cols = list(dict.fromkeys([col for col in df.columns if ":" not in col]))
    process_names = set([col.split(":")[0] for col in df.columns if ":" in col])
    split_dfs = {}
    for process in process_names:
        process_specific_cols = [col for col in df.columns if col.startswith(f"{process}:")]
        current_all_cols = common_cols + process_specific_cols
        process_df = df.loc[:, ~df.columns.duplicated()][current_all_cols].copy()
        process_df.columns = [col.split(":")[-1] if ":" in col else col for col in process_df.columns]
        process_df = process_df.dropna(axis=1, how="all")
        split_dfs[process] = process_df
    return split_dfs


def filt_specific_name(df, que):
    selected_cols = df.columns[df.columns.str.contains(que, case=False)]
    return df[selected_cols]


def extract_batch_logic(s):
    match = re.search(r"(\d{4})$", str(s))
    return int(match.group(1)) if match else 0


def extract_number(s):
    match = re.search(r"\d+", str(s))
    return int(match.group()) if match else 0


def smooth_process_data(df, target_cols, id_cols=["BatchID"], method="loess", frac=0.3, span=10):
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
                temp_series = pd.Series(res_y, index=df.index)
                smoothed_df[col] = temp_series.interpolate(limit_direction="both")
            else:
                smoothed_df[col] = y
        elif method.lower() == "ewma":
            smoothed_df[col] = df[col].ewm(span=span, adjust=False).mean()
    return smoothed_df


def plot_indexed_lineplots(df, batch_col="BatchID", cols_per_row=3):
    sorted_df = df.copy()
    sorted_df["temp_sort_val"] = sorted_df[batch_col].apply(extract_number)
    sorted_df = sorted_df.sort_values("temp_sort_val").reset_index(drop=True)
    sorted_df["Sequence_Index"] = sorted_df.index + 1

    exclude_cols = [batch_col, "temp_sort_val", "Sequence_Index"]
    numeric_cols = [c for c in sorted_df.select_dtypes(include=["number"]).columns if c not in exclude_cols]

    if not numeric_cols:
        st.warning("沒有找到可繪圖的數值型欄位。")
        return None

    num_rows = math.ceil(len(numeric_cols) / cols_per_row)
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row * 5, num_rows * 4))
    axes = np.array(axes).flatten()
    sns.set_style("whitegrid")

    for i, col in enumerate(numeric_cols):
        sns.lineplot(data=sorted_df, x="Sequence_Index", y=col, marker="o",
                     color="royalblue", linewidth=1.5, ax=axes[i])
        wrapped_title = "\n".join(textwrap.wrap(col, width=30))
        axes[i].set_title(wrapped_title, fontsize=9, pad=8)
        axes[i].set_xlabel("Batch Sequence")
        axes[i].set_ylabel("Value")

    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig


def plot_clean_lineplots(df, batch_col="BatchID", cols_per_row=3):
    temp_df = df.copy()
    temp_df["sort_key"] = temp_df[batch_col].apply(extract_batch_logic)
    temp_df = temp_df.sort_values("sort_key")

    numeric_cols = temp_df.select_dtypes(include=["number"]).columns.tolist()
    for c in ["sort_key", "batch_num"]:
        if c in numeric_cols:
            numeric_cols.remove(c)

    if not numeric_cols:
        return None

    num_rows = math.ceil(len(numeric_cols) / cols_per_row)
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row * 5, num_rows * 4))
    axes = np.array(axes).flatten()
    sns.set_style("whitegrid")

    for i, col in enumerate(numeric_cols):
        sns.lineplot(data=temp_df, x="sort_key", y=col, marker="o", color="teal", ax=axes[i])
        axes[i].xaxis.set_major_formatter(plt.FormatStrFormatter("%d"))
        wrapped_title = "\n".join(textwrap.wrap(col, width=30))
        axes[i].set_title(wrapped_title, fontsize=9, pad=8)
        axes[i].set_xlabel("Batch (YYNN)")
        axes[i].set_ylabel("Value")

    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    return fig


def clean_process_features_with_log(df, id_col="BatchID", protected_cols=None):
    df = df.loc[:, ~df.columns.duplicated()]
    new_df = df.copy().reset_index(drop=True)
    drop_log = []
    if protected_cols is None:
        protected_cols = []
    whitelist = set([id_col] + protected_cols)

    # Rule A: keyword filter
    target_keywords = ["Verification Result", "No (na)"]
    to_drop_kw = [
        c for c in new_df.columns if (
            any(kw in c for kw in target_keywords) and
            not c.strip().lower().endswith("(times)")
        ) and c not in whitelist
    ]
    for c in to_drop_kw:
        drop_log.append({"Column": c, "Reason": "Keyword Filter"})
    new_df = new_df.drop(columns=to_drop_kw)

    def clean_col_name(name):
        return re.sub(r"\s?\(.*\)$", "", name).strip()

    # Rule B: paired difference
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

    # Rule C: numbered columns → average
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
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep_cols = []
    dropped_info = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 5:
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

    final_cols = keep_cols + df.select_dtypes(exclude=[np.number]).columns.tolist()
    return df[final_cols], dropped_info


def analyze_correlation(df, target_col, method="pearson", top_n=10):
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    if target_col not in numeric_df.columns:
        return None
    corr_matrix = numeric_df.corr(method=method)
    corr_series = corr_matrix[target_col].drop(target_col)
    corr_rank = pd.DataFrame({
        "Feature": corr_series.index,
        "Correlation": corr_series.values,
        "Abs_Correlation": corr_series.abs().values,
    }).sort_values(by="Abs_Correlation", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_corr = corr_rank.head(top_n)
    sns.barplot(data=top_corr, x="Correlation", y="Feature", palette="vlag", ax=ax)
    ax.axvline(0, color="black", linestyle="-", linewidth=1)
    ax.set_title(f"Top {top_n} Features Correlated with\n{target_col}", fontsize=12)
    ax.set_xlabel(f"{method.capitalize()} Correlation Coefficient")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    return fig, corr_rank[["Feature", "Correlation"]].reset_index(drop=True)


def missing_col(df):
    na_counts = df.isnull().sum()
    total_batches = len(df)
    na_ratio = (na_counts / total_batches) * 100
    mask = na_counts > 0
    filtered_counts = na_counts[mask].sort_values(ascending=False)
    filtered_ratio = na_ratio[mask].sort_values(ascending=False)
    missing_summary = pd.concat([filtered_counts, filtered_ratio], axis=1)
    missing_summary.columns = ["Missing Count", "Missing Ratio (%)"]
    return missing_summary


def rf_feature_importance(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, max_features=5, max_depth=5,
                                min_samples_leaf=4, random_state=42)
    rf.fit(X_train, y_train)
    perm_results = permutation_importance(rf, X_train, y_train, n_repeats=10, random_state=42)
    perm_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Perm_Importance": perm_results.importances_mean,
    }).sort_values(by="Perm_Importance", ascending=False)
    return perm_importance, rf


def pls_feature_importance(X_train, y_train, n_components=3):
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)

    def calculate_vip(model):
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
            vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
        return vips

    vip_scores = calculate_vip(pls)
    vip_df = pd.DataFrame({
        "Feature": X_train.columns,
        "VIP": vip_scores,
    }).sort_values(by="VIP", ascending=False)
    return vip_df


# ═══════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════════════════
for key in ["raw_df", "dfs_dict", "selected_process_df", "clean_df",
            "X", "y", "label", "x_scaled"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dna-helix.png", width=60)
    st.title("🧬 Bioprocess Analytics")
    st.markdown("---")
    st.markdown("### 📁 資料載入")
    uploaded_file = st.file_uploader("上傳 CSV 資料檔", type=["csv"])

    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            # Rename BatchID column
            batch_col_candidates = [c for c in raw_df.columns if "BatchID" in c]
            if batch_col_candidates:
                raw_df = raw_df.rename(columns={batch_col_candidates[0]: "BatchID"})
            # Convert to numeric
            exclude_cols = ["BatchID"]
            cols_to_convert = raw_df.columns.difference(exclude_cols)
            raw_df[cols_to_convert] = raw_df[cols_to_convert].apply(pd.to_numeric, errors="coerce")
            st.session_state["raw_df"] = raw_df
            st.session_state["dfs_dict"] = split_process_df(raw_df)
            st.success(f"✅ 載入成功！{raw_df.shape[0]} 筆 × {raw_df.shape[1]} 欄")
        except Exception as e:
            st.error(f"載入失敗：{e}")

    st.markdown("---")
    if st.session_state["dfs_dict"]:
        st.markdown("### ⚙️ 製程步驟選擇")
        process_list = list(st.session_state["dfs_dict"].keys())
        selected_process = st.selectbox("選擇製程步驟", process_list)
        st.session_state["selected_process"] = selected_process
        st.session_state["selected_process_df"] = st.session_state["dfs_dict"][selected_process]

    st.markdown("---")
    st.caption("Bioprocess Analytics v1.0")


# ═══════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════
st.title("🧬 Bioprocess Data Analysis Platform")

if st.session_state["raw_df"] is None:
    st.markdown("""
    <div class="info-box">
    👈 請先在左側上傳 <b>CSV 資料檔</b>（raw_data.csv）開始分析。
    </div>
    """, unsafe_allow_html=True)
    st.stop()

raw_df = st.session_state["raw_df"]
dfs_dict = st.session_state["dfs_dict"]
selected_process_df = st.session_state.get("selected_process_df")
selected_process = st.session_state.get("selected_process", "")

# ── Tabs ──────────────────────────────────────────────────
tabs = st.tabs([
    "📊 資料總覽",
    "📈 趨勢圖",
    "🔧 特徵工程",
    "🔍 缺失值分析",
    "🔗 相關性分析",
    "🧩 PCA 分析",
    "🌲 特徵重要性",
])

# ─────────────────────────────────────────────────────────
# TAB 0: 資料總覽
# ─────────────────────────────────────────────────────────
with tabs[0]:
    st.header("資料總覽")

    col1, col2, col3 = st.columns(3)
    col1.metric("總批次數", raw_df.shape[0])
    col2.metric("總欄位數", raw_df.shape[1])
    col3.metric("製程步驟數", len(dfs_dict))

    st.markdown("#### 製程步驟欄位統計")
    step_df = process_step_count(raw_df)
    st.dataframe(step_df, width="stretch", hide_index=True)

    st.markdown("#### 原始資料預覽")
    st.dataframe(raw_df.head(10), width="stretch")

    if selected_process_df is not None:
        st.markdown(f"#### 已選製程：`{selected_process}` — 欄位預覽")
        st.dataframe(selected_process_df.head(10), width="stretch")

# ─────────────────────────────────────────────────────────
# TAB 1: 趨勢圖
# ─────────────────────────────────────────────────────────
with tabs[1]:
    st.header("趨勢圖")
    if selected_process_df is None:
        st.info("請先在側欄選擇製程步驟。")
    else:
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            keyword = st.text_input("欄位關鍵字篩選（留空 = 全部）", "")
        with col_b:
            smooth_method = st.selectbox("平滑方法", ["loess", "ewma", "none"])
        with col_c:
            cols_per_row = st.slider("每列圖數", 1, 5, 3)

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

# ─────────────────────────────────────────────────────────
# TAB 2: 特徵工程
# ─────────────────────────────────────────────────────────
with tabs[2]:
    st.header("特徵工程 & 清理")
    if selected_process_df is None:
        st.info("請先在側欄選擇製程步驟。")
    else:
        st.markdown("""
        **自動執行以下規則：**
        - 🗑️ 過濾含有 `Verification Result` / `No (na)` 關鍵字的欄位
        - ➕ 配對 Max/Min、After/Before、End/Start → 計算差值
        - 🔢 數字編號欄位（如 _1、_2）→ 取平均後合併
        """)

        if st.button("🔧 執行特徵工程", key="run_fe"):
            with st.spinner("處理中..."):
                clean_df, drop_log = clean_process_features_with_log(
                    selected_process_df, id_col="BatchID"
                )
                st.session_state["clean_df"] = clean_df
                st.success(f"✅ 完成！從 {selected_process_df.shape[1]} 欄 → {clean_df.shape[1]} 欄")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 清理後資料預覽")
                    st.dataframe(clean_df.head(10), width="stretch")
                with col2:
                    st.markdown("#### 刪除/合併記錄")
                    st.dataframe(drop_log, width="stretch", hide_index=True)

        if st.session_state["clean_df"] is not None:
            clean_df = st.session_state["clean_df"]
            st.markdown("---")
            st.markdown("#### 📉 統計篩選（移除低資訊量欄位）")
            c1, c2, c3 = st.columns(3)
            cv_thresh = c1.slider("CV 門檻（低於此值剔除）", 0.0, 0.1, 0.01, 0.001, format="%.3f")
            jump_thresh = c2.slider("Jump Ratio 門檻（高於此值剔除）", 0.1, 1.0, 0.5, 0.05)
            acf_thresh = c3.slider("ACF 門檻（低於此值剔除）", 0.0, 0.5, 0.2, 0.05)

            if st.button("📉 執行統計篩選", key="run_stat_filter"):
                with st.spinner("篩選中..."):
                    filtered_df, dropped_info = filter_columns_by_stats(
                        clean_df, cv_threshold=cv_thresh,
                        jump_ratio_threshold=jump_thresh,
                        acf_threshold=acf_thresh
                    )
                    if "BatchID" in clean_df.columns and "BatchID" not in filtered_df.columns:
                        filtered_df.insert(0, "BatchID", clean_df["BatchID"])

                    st.session_state["clean_df"] = filtered_df
                    st.success(f"✅ 剔除 {len(dropped_info)} 個欄位 → 剩餘 {filtered_df.shape[1]} 欄")

                    if dropped_info:
                        drop_df = pd.DataFrame(
                            [(k, v) for k, v in dropped_info.items()],
                            columns=["Column", "Reason"]
                        )
                        st.dataframe(drop_df, width="stretch", hide_index=True)

# ─────────────────────────────────────────────────────────
# TAB 3: 缺失值分析
# ─────────────────────────────────────────────────────────
with tabs[3]:
    st.header("缺失值分析")
    _cd = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df
    if work_df is None:
        st.info("請先在側欄選擇製程步驟，或執行特徵工程。")
    else:
        summary_df = missing_col(work_df)
        if summary_df.empty:
            st.success("🎉 無缺失值！")
        else:
            st.metric("含缺失值的欄位數", len(summary_df))
            st.dataframe(summary_df.style.background_gradient(cmap="Reds", subset=["Missing Ratio (%)"]),
                         width="stretch")

            # Missing heatmap
            st.markdown("#### 缺失值熱圖")
            fig, ax = plt.subplots(figsize=(14, 4))
            missing_matrix = work_df[summary_df.index].isnull().T
            sns.heatmap(missing_matrix, cmap="Reds", cbar=False, ax=ax,
                        yticklabels=[c[:40] for c in summary_df.index])
            ax.set_xlabel("Sample Index")
            ax.set_title("Missing Value Pattern")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.markdown("#### 🗑️ 手動移除批次")
        if "BatchID" in work_df.columns:
            batch_ids = work_df["BatchID"].tolist()
            drop_batches = st.multiselect("選擇要移除的 BatchID", batch_ids)
            drop_cols_ui = st.text_input("要移除的欄位名稱（逗號分隔）", "")

            if st.button("🗑️ 執行移除", key="drop_rows"):
                filtered = work_df.copy()
                if drop_batches:
                    filtered = filtered[~filtered["BatchID"].isin(drop_batches)]
                if drop_cols_ui.strip():
                    cols_to_drop = [c.strip() for c in drop_cols_ui.split(",") if c.strip() in filtered.columns]
                    filtered = filtered.drop(columns=cols_to_drop)
                st.session_state["clean_df"] = filtered
                st.success(f"✅ 移除後：{filtered.shape[0]} 筆 × {filtered.shape[1]} 欄")
                st.dataframe(filtered.head(), width="stretch")

# ─────────────────────────────────────────────────────────
# TAB 4: 相關性分析
# ─────────────────────────────────────────────────────────
with tabs[4]:
    st.header("相關性分析")
    _cd = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df
    if work_df is None:
        st.info("請先在側欄選擇製程步驟。")
    else:
        numeric_cols = work_df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            st.warning("無數值型欄位。")
        else:
            col1, col2, col3 = st.columns(3)
            target_col = col1.selectbox("目標欄位（Y）", numeric_cols)
            method = col2.selectbox("相關係數方法", ["pearson", "spearman"])
            top_n = col3.slider("顯示前 N 個特徵", 5, min(50, len(numeric_cols)), 15)

            if st.button("🔗 計算相關性", key="run_corr"):
                with st.spinner("計算中..."):
                    result = analyze_correlation(work_df, target_col, method=method, top_n=top_n)
                    if result:
                        fig, corr_rank = result
                        st.pyplot(fig)
                        plt.close()
                        st.markdown("#### 相關係數排行")
                        st.dataframe(
                            corr_rank.style.background_gradient(
                                cmap="RdBu_r", subset=["Correlation"], vmin=-1, vmax=1),
                            width="stretch", hide_index=True
                        )
                        # Store for other tabs
                        st.session_state["target_col"] = target_col
                        st.session_state["corr_rank"] = corr_rank

# ─────────────────────────────────────────────────────────
# TAB 5: PCA 分析
# ─────────────────────────────────────────────────────────
with tabs[5]:
    st.header("PCA 主成分分析")
    _cd = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df
    if work_df is None:
        st.info("請先在側欄選擇製程步驟。")
    else:
        target_col_pca = st.session_state.get("target_col")
        if target_col_pca is None:
            numeric_options = work_df.select_dtypes(include=["number"]).columns.tolist()
            target_col_pca = st.selectbox("請先選擇目標欄位（用於排除）", numeric_options)

        col1, col2 = st.columns(2)
        n_components = col1.slider("最大主成分數", 2, 15, 10)
        alpha_pca = col2.slider("異常值顯著水準 α", 0.01, 0.1, 0.05, 0.01)

        if st.button("🧩 執行 PCA", key="run_pca"):
            try:
                exclude = ["BatchID", target_col_pca] if target_col_pca else ["BatchID"]
                X_pca = work_df.drop(columns=[c for c in exclude if c in work_df.columns],
                                     errors="ignore")
                X_pca = X_pca.select_dtypes(include=["number"]).dropna(axis=1)

                scaler = StandardScaler()
                x_scaled = scaler.fit_transform(X_pca)

                with st.spinner("PCA 計算中..."):
                    model = pca(n_components=n_components, alpha=alpha_pca)
                    results = model.fit_transform(x_scaled)

                col1, col2, col3 = st.columns(3)
                ev = model.results["explained_var"]
                col1.metric("PC1 累計解釋變異", f"{ev[0]*100:.1f}%")
                col2.metric("PC2 累計解釋變異", f"{ev[1]*100:.1f}%")
                col3.metric(f"PC{len(ev)} 累計解釋變異", f"{ev[-1]*100:.1f}%")

                # Scree plot
                fig_scree, ax_s = model.plot()
                st.pyplot(fig_scree)
                plt.close()

                # Scatter
                fig_sc, ax_sc = model.scatter()
                st.pyplot(fig_sc)
                plt.close()

                # Biplot
                st.markdown("#### Biplot（PC1 vs PC2）")
                fig_bi, ax_bi = model.biplot(n_feat=6, PC=[0, 1], legend=True, SPE=True, HT2=True)
                st.pyplot(fig_bi)
                plt.close()

                st.markdown("#### Top Features per PC")
                topfeat = model.results.get("topfeat")
                if topfeat is not None:
                    # Map feature index to name
                    feat_names = X_pca.columns.tolist()
                    def safe_map(x):
                        try:
                            idx = int(x)
                            return feat_names[idx] if 0 <= idx < len(feat_names) else str(x)
                        except:
                            return str(x)
                    topfeat = topfeat.copy()
                    topfeat["feature_name"] = topfeat["feature"].apply(safe_map)
                    st.dataframe(topfeat, width="stretch", hide_index=True)

            except Exception as e:
                st.error(f"PCA 執行失敗：{e}")

# ─────────────────────────────────────────────────────────
# TAB 6: 特徵重要性
# ─────────────────────────────────────────────────────────
with tabs[6]:
    st.header("特徵重要性分析")
    _cd = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df
    if work_df is None:
        st.info("請先在側欄選擇製程步驟。")
    else:
        numeric_cols = work_df.select_dtypes(include=["number"]).columns.tolist()
        target_col_fi = st.selectbox("目標欄位（Y）", numeric_cols, key="fi_target")

        method_fi = st.radio("分析方法", ["Random Forest (Permutation Importance)", "PLS-VIP"], horizontal=True)

        col1, col2 = st.columns(2)
        top_n_fi = col1.slider("顯示前 N 個特徵", 5, 30, 15, key="fi_topn")
        if "PLS" in method_fi:
            n_pls = col2.slider("PLS 主成分數", 1, 10, 3)

        if st.button("🌲 執行特徵重要性分析", key="run_fi"):
            try:
                exclude = ["BatchID", target_col_fi]
                X_fi = work_df.drop(columns=[c for c in exclude if c in work_df.columns], errors="ignore")
                X_fi = X_fi.select_dtypes(include=["number"])
                y_fi = work_df[target_col_fi]

                valid_idx = y_fi.notna() & X_fi.notna().all(axis=1)
                X_fi = X_fi[valid_idx].reset_index(drop=True)
                y_fi = y_fi[valid_idx].reset_index(drop=True)

                with st.spinner("模型訓練中..."):
                    if "Random Forest" in method_fi:
                        importance_df, rf_model = rf_feature_importance(X_fi, y_fi)
                        importance_col = "Perm_Importance"
                    else:
                        importance_df = pls_feature_importance(X_fi, y_fi, n_components=n_pls)
                        importance_col = "VIP"

                top_df = importance_df.head(top_n_fi)

                fig, ax = plt.subplots(figsize=(10, max(4, top_n_fi * 0.4)))
                colors = ["#2e86ab" if v > 0 else "#e84855" for v in top_df[importance_col]]
                ax.barh(top_df["Feature"], top_df[importance_col], color=colors, alpha=0.85)
                ax.axvline(0, color="black", linewidth=1)
                ax.set_title(f"Top {top_n_fi} Features — {method_fi}", fontsize=13)
                ax.set_xlabel(importance_col)
                ax.invert_yaxis()
                ax.grid(axis="x", linestyle="--", alpha=0.5)
                plt.tight_layout()

                st.pyplot(fig)
                plt.close()

                st.markdown("#### 重要性排行表")
                st.dataframe(importance_df.reset_index(drop=True).style.background_gradient(
                    cmap="Blues", subset=[importance_col]),
                    width="stretch", hide_index=True
                )

                # PLS cross-validation curve
                if "PLS" in method_fi:
                    st.markdown("#### PLS 交叉驗證 MSE（選擇最佳主成分數）")
                    mse_list = []
                    comp_range = range(1, min(11, X_fi.shape[1]))
                    for n in comp_range:
                        pls_cv = PLSRegression(n_components=n)
                        y_cv = cross_val_predict(pls_cv, X_fi, y_fi, cv=5)
                        mse_list.append(mean_squared_error(y_fi, y_cv))
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.plot(list(comp_range), mse_list, marker="o", color="#2e86ab")
                    ax2.set_xlabel("Number of Components")
                    ax2.set_ylabel("MSE (Cross-Validation)")
                    ax2.set_title("PLS CV MSE — Elbow Curve")
                    ax2.grid(linestyle="--", alpha=0.5)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()

            except Exception as e:
                st.error(f"分析失敗：{e}")
