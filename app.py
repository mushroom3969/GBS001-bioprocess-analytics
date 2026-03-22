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
    "📚 文獻佐證分析",
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
        numeric_options_pca = work_df.select_dtypes(include=["number"]).columns.tolist()
        stored_target = st.session_state.get("target_col")
        default_idx = numeric_options_pca.index(stored_target) if stored_target in numeric_options_pca else 0

        with st.expander("⚙️ PCA 設定", expanded=True):
            pc1, pc2, pc3 = st.columns(3)
            target_col_pca = pc1.selectbox("排除的目標欄位（Y）", numeric_options_pca, index=default_idx, key="pca_target")
            n_components   = pc2.slider("最大主成分數", 2, min(15, len(numeric_options_pca)-1), 5)
            alpha_pca      = pc3.select_slider("Hotelling T² 顯著水準 α", [0.01, 0.05, 0.10], value=0.05)

        if st.button("🧩 執行 PCA", key="run_pca"):
            try:
                exclude_pca = [c for c in ["BatchID", target_col_pca] if c in work_df.columns]
                X_pca = work_df.drop(columns=exclude_pca, errors="ignore")
                X_pca = X_pca.select_dtypes(include=["number"]).dropna(axis=1)

                if X_pca.shape[1] < 2:
                    st.error("有效欄位不足 2 個，無法執行 PCA。請先完成特徵工程或調整篩選條件。")
                else:
                    label_pca = work_df["BatchID"].values if "BatchID" in work_df.columns else np.arange(len(work_df))
                    scaler_pca = StandardScaler()
                    x_scaled_pca = scaler_pca.fit_transform(X_pca)
                    feat_names_pca = X_pca.columns.tolist()
                    n_comp_actual  = min(n_components, X_pca.shape[1] - 1)

                    with st.spinner("PCA 計算中..."):
                        model_pca = pca(n_components=n_comp_actual, alpha=alpha_pca,
                                        detect_outliers=["ht2", "spe"])
                        results_pca = model_pca.fit_transform(x_scaled_pca)

                    # ── 儲存到 session ──
                    st.session_state["pca_model"]   = model_pca
                    st.session_state["pca_results"] = results_pca
                    st.session_state["pca_X"]       = X_pca
                    st.session_state["pca_x_scaled"]= x_scaled_pca
                    st.session_state["pca_labels"]  = label_pca
                    st.session_state["pca_feat"]    = feat_names_pca
                    st.success("✅ PCA 完成！")

            except Exception as e:
                st.error(f"PCA 執行失敗：{e}")

        # ── 顯示結果（只要 session 有資料就顯示）──
        if st.session_state.get("pca_model") is not None:
            model_pca    = st.session_state["pca_model"]
            results_pca  = st.session_state["pca_results"]
            X_pca        = st.session_state["pca_X"]
            x_scaled_pca = st.session_state["pca_x_scaled"]
            label_pca    = st.session_state["pca_labels"]
            feat_names_pca = st.session_state["pca_feat"]

            ev  = model_pca.results["explained_var"]
            vr  = model_pca.results["variance_ratio"]
            scores   = results_pca["PC"].values
            loadings = model_pca.results["loadings"].values
            n_pc     = scores.shape[1]

            # ── Metrics ──
            cols_m = st.columns(min(n_pc, 5))
            for i, c in enumerate(cols_m):
                c.metric(f"PC{i+1} 累計解釋", f"{ev[i]*100:.1f}%",
                         delta=f"+{vr[i]*100:.1f}%")

            pca_subtabs = st.tabs(["📊 Scree & Scatter", "🔵 Biplot",
                                   "🚨 Hotelling T² 異常偵測", "🔬 單筆貢獻分析"])

            # ── Subtab 0: Scree + Scatter ──
            with pca_subtabs[0]:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Scree Plot**")
                    fig_sc, _ = model_pca.plot()
                    st.pyplot(fig_sc); plt.close()
                with c2:
                    st.markdown("**Score Scatter (PC1 vs PC2)**")
                    fig_s2, _ = model_pca.scatter(SPE=True, HT2=True)
                    st.pyplot(fig_s2); plt.close()

            # ── Subtab 1: Biplot ──
            with pca_subtabs[1]:
                bp1, bp2 = st.columns(2)
                n_feat_bi = bp1.slider("顯示載荷向量數", 3, min(15, len(feat_names_pca)), 6, key="bi_nfeat")
                pc_options = list(range(n_pc))
                pc_x = bp2.selectbox("X 軸 PC", pc_options, index=0, format_func=lambda x: f"PC{x+1}", key="bi_pcx")
                pc_y = bp2.selectbox("Y 軸 PC", pc_options, index=min(1, n_pc-1), format_func=lambda x: f"PC{x+1}", key="bi_pcy")
                fig_bi, _ = model_pca.biplot(n_feat=n_feat_bi, PC=[pc_x, pc_y], legend=True, SPE=True, HT2=True)
                st.pyplot(fig_bi); plt.close()

                st.markdown("**Top Features per PC**")
                topfeat = model_pca.results.get("topfeat")
                if topfeat is not None:
                    tf = topfeat.copy()
                    def safe_map_feat(x):
                        try:
                            idx = int(x)
                            return feat_names_pca[idx] if 0 <= idx < len(feat_names_pca) else str(x)
                        except:
                            return str(x)
                    tf["feature_name"] = tf["feature"].apply(safe_map_feat)
                    st.dataframe(tf, width="stretch", hide_index=True)

            # ── Subtab 2: Hotelling T² 異常偵測 ──
            with pca_subtabs[2]:
                st.markdown("#### Hotelling T² — 各 Batch 異常程度")

                # 計算每個樣本的 T² 值
                ht2_vals = np.sum((scores ** 2) / ev, axis=1)

                # 計算三個信心水準的閾值
                n_obs, p = x_scaled_pca.shape[0], n_pc
                from scipy.stats import f as f_dist
                def ht2_threshold(alpha, n, p):
                    f_crit = f_dist.ppf(1 - alpha, p, n - p)
                    return (p * (n - 1) * (n + 1)) / (n * (n - p)) * f_crit

                thres_68 = ht2_threshold(0.32, n_obs, p)
                thres_95 = ht2_threshold(0.05, n_obs, p)
                thres_99 = ht2_threshold(0.01, n_obs, p)

                # 建立結果表
                ht2_df = pd.DataFrame({
                    "Batch": label_pca,
                    "T²": ht2_vals.round(3),
                })
                def classify_ht2(v):
                    if v > thres_99: return "🔴 >99%"
                    elif v > thres_95: return "🟠 >95%"
                    elif v > thres_68: return "🟡 >68%"
                    else: return "🟢 Normal"
                ht2_df["Status"] = ht2_df["T²"].apply(classify_ht2)
                ht2_df = ht2_df.sort_values("T²", ascending=False).reset_index(drop=True)

                # 指標
                m1, m2, m3 = st.columns(3)
                m1.metric("超過 68% 閾值", f"{(ht2_vals > thres_68).sum()} 批")
                m2.metric("超過 95% 閾值", f"{(ht2_vals > thres_95).sum()} 批")
                m3.metric("超過 99% 閾值", f"{(ht2_vals > thres_99).sum()} 批")

                # 圖
                fig_ht, ax_ht = plt.subplots(figsize=(14, 5))
                idx_sorted = np.argsort([extract_number(str(b)) for b in label_pca])
                x_plot = np.arange(len(label_pca))
                bar_colors = []
                for v in ht2_vals[idx_sorted]:
                    if v > thres_99: bar_colors.append("#e84855")
                    elif v > thres_95: bar_colors.append("#f4a261")
                    elif v > thres_68: bar_colors.append("#e9c46a")
                    else: bar_colors.append("#2e86ab")

                ax_ht.bar(x_plot, ht2_vals[idx_sorted], color=bar_colors, alpha=0.85, width=0.7)
                ax_ht.axhline(thres_68, color="#e9c46a", linestyle="--", lw=1.5, label=f"68% ({thres_68:.1f})")
                ax_ht.axhline(thres_95, color="#f4a261", linestyle="--", lw=1.5, label=f"95% ({thres_95:.1f})")
                ax_ht.axhline(thres_99, color="#e84855", linestyle="--", lw=1.5, label=f"99% ({thres_99:.1f})")
                ax_ht.set_xticks(x_plot)
                ax_ht.set_xticklabels([str(label_pca[i])[-6:] for i in idx_sorted], rotation=90, fontsize=7)
                ax_ht.set_ylabel("Hotelling T² Value")
                ax_ht.set_title("Hotelling T² per Batch (sorted by time)")
                ax_ht.legend(title="Confidence Level")
                ax_ht.grid(axis="y", linestyle="--", alpha=0.4)
                plt.tight_layout()
                st.pyplot(fig_ht); plt.close()

                st.markdown("**Batch 排行（T² 由大到小）**")
                st.dataframe(ht2_df, width="stretch", hide_index=True)

            # ── Subtab 3: 單筆貢獻分析 ──
            with pca_subtabs[3]:
                st.markdown("#### 選擇要分析的 Batch")
                batch_options = [str(b) for b in label_pca]
                sel_batch = st.selectbox("選擇 Batch", batch_options, key="pca_sel_batch")
                sample_i  = batch_options.index(sel_batch)

                view_mode = st.radio("分析模式",
                    ["所有 PC 的特徵貢獻（總 T²）", "單一 PC 的特徵貢獻"],
                    horizontal=True, key="pca_view_mode")

                top_n_contrib = st.slider("顯示前 N 個特徵", 5, min(30, len(feat_names_pca)), 15, key="pca_top_contrib")

                if view_mode == "所有 PC 的特徵貢獻（總 T²）":
                    # 全 PC 貢獻
                    contributions = np.zeros(loadings.shape[1])
                    for a in range(n_pc):
                        contributions += (scores[sample_i, a] / ev[a]) * loadings[a, :] * x_scaled_pca[sample_i, :]

                    df_contrib = pd.DataFrame({
                        "Feature": feat_names_pca,
                        "Contribution": contributions,
                    }).reindex(pd.Series(contributions).abs().sort_values(ascending=False).index)
                    df_contrib = df_contrib.reset_index(drop=True).head(top_n_contrib)

                    fig_c, ax_c = plt.subplots(figsize=(12, max(5, top_n_contrib * 0.4)))
                    colors_c = ["#e84855" if v > 0 else "#2e86ab" for v in df_contrib["Contribution"]]
                    ax_c.barh(df_contrib["Feature"], df_contrib["Contribution"], color=colors_c, alpha=0.85)
                    ax_c.axvline(0, color="black", lw=1)
                    ax_c.set_title(f"Hotelling T² Total Contribution — {sel_batch}", fontsize=13)
                    ax_c.set_xlabel("Contribution")
                    ax_c.invert_yaxis()
                    ax_c.grid(axis="x", linestyle="--", alpha=0.5)
                    plt.tight_layout()
                    st.pyplot(fig_c); plt.close()
                    st.dataframe(df_contrib, width="stretch", hide_index=True)

                    # 各 PC 的 T² 分解圖
                    st.markdown("**各 PC 對該 Batch T² 的貢獻分解**")
                    t2_per_pc = (scores[sample_i, :] ** 2) / ev
                    fig_pc_bar, ax_pc = plt.subplots(figsize=(8, 4))
                    pc_labels_bar = [f"PC{j+1}" for j in range(n_pc)]
                    ax_pc.bar(pc_labels_bar, t2_per_pc, color="#2e86ab", alpha=0.8)
                    for bar, val in zip(ax_pc.patches, t2_per_pc):
                        ax_pc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f"{val:.2f}", ha="center", va="bottom", fontsize=9)
                    ax_pc.set_ylabel("T² per PC")
                    ax_pc.set_title(f"PC-wise T² Decomposition — {sel_batch}")
                    ax_pc.grid(axis="y", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig_pc_bar); plt.close()

                else:
                    # 單一 PC 貢獻
                    sel_pc = st.selectbox("選擇 PC", list(range(n_pc)),
                                          format_func=lambda x: f"PC{x+1}", key="pca_sel_pc")
                    pc_contrib = (scores[sample_i, sel_pc] / ev[sel_pc]) * loadings[sel_pc, :] * x_scaled_pca[sample_i, :]
                    df_pc = pd.DataFrame({
                        "Feature": feat_names_pca,
                        "Contribution": pc_contrib,
                    }).reindex(pd.Series(pc_contrib).abs().sort_values(ascending=False).index)
                    df_pc = df_pc.reset_index(drop=True).head(top_n_contrib)

                    fig_pc, ax_pc2 = plt.subplots(figsize=(12, max(5, top_n_contrib * 0.4)))
                    colors_pc = ["#e84855" if v > 0 else "#2e86ab" for v in df_pc["Contribution"]]
                    ax_pc2.barh(df_pc["Feature"], df_pc["Contribution"], color=colors_pc, alpha=0.85)
                    ax_pc2.axvline(0, color="black", lw=1)
                    ax_pc2.set_title(f"PC{sel_pc+1} Feature Contribution — {sel_batch}", fontsize=13)
                    ax_pc2.set_xlabel("Contribution")
                    ax_pc2.invert_yaxis()
                    ax_pc2.grid(axis="x", linestyle="--", alpha=0.5)
                    plt.tight_layout()
                    st.pyplot(fig_pc); plt.close()
                    st.dataframe(df_pc, width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────
# TAB 6: 特徵重要性 + SHAP + PLS
# ─────────────────────────────────────────────────────────
with tabs[6]:
    st.header("特徵重要性分析")
    _cd = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df
    if work_df is None:
        st.info("請先在側欄選擇製程步驟。")
    else:
        numeric_cols = work_df.select_dtypes(include=["number"]).columns.tolist()
        stored_target_fi = st.session_state.get("target_col")
        default_fi = numeric_cols.index(stored_target_fi) if stored_target_fi in numeric_cols else 0
        target_col_fi = st.selectbox("目標欄位（Y）", numeric_cols, index=default_fi, key="fi_target")
        top_n_fi = st.slider("顯示前 N 個特徵", 5, 30, 15, key="fi_topn")

        if st.button("🚀 準備資料 & 訓練模型", key="run_fi_prepare"):
            try:
                exclude_fi = [c for c in ["BatchID", target_col_fi] if c in work_df.columns]
                X_fi = work_df.drop(columns=exclude_fi, errors="ignore").select_dtypes(include=["number"])
                y_fi = work_df[target_col_fi]
                valid_idx = y_fi.notna() & X_fi.notna().all(axis=1)
                X_fi = X_fi[valid_idx].reset_index(drop=True)
                y_fi = y_fi[valid_idx].reset_index(drop=True)

                with st.spinner("Random Forest 訓練中..."):
                    rf = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                               max_depth=5, min_samples_leaf=4, random_state=42)
                    rf.fit(X_fi, y_fi)
                    perm = permutation_importance(rf, X_fi, y_fi, n_repeats=15, random_state=42)
                    perm_df = pd.DataFrame({
                        "Feature": X_fi.columns,
                        "Perm_Importance": perm.importances_mean,
                        "Std": perm.importances_std,
                    }).sort_values("Perm_Importance", ascending=False).reset_index(drop=True)

                st.session_state["fi_X"]       = X_fi
                st.session_state["fi_y"]       = y_fi
                st.session_state["fi_rf"]      = rf
                st.session_state["fi_perm_df"] = perm_df
                st.session_state["fi_target_col"] = target_col_fi
                st.success("✅ 模型訓練完成！")
            except Exception as e:
                st.error(f"訓練失敗：{e}")

        if st.session_state.get("fi_rf") is not None:
            X_fi    = st.session_state["fi_X"]
            y_fi    = st.session_state["fi_y"]
            rf      = st.session_state["fi_rf"]
            perm_df = st.session_state["fi_perm_df"]

            fi_subtabs = st.tabs(["🌲 RF 重要性", "🔮 SHAP 分析", "📐 PLS-VIP"])

            # ── RF ──
            with fi_subtabs[0]:
                st.markdown("#### Random Forest — Permutation Importance")
                top_perm = perm_df.head(top_n_fi)
                fig_rf, ax_rf = plt.subplots(figsize=(10, max(5, top_n_fi * 0.45)))
                colors_rf = ["#2e86ab" if v >= 0 else "#e84855" for v in top_perm["Perm_Importance"]]
                bars = ax_rf.barh(top_perm["Feature"], top_perm["Perm_Importance"],
                                   xerr=top_perm["Std"], color=colors_rf, alpha=0.85,
                                   error_kw={"ecolor": "gray", "capsize": 3})
                ax_rf.axvline(0, color="black", lw=1)
                ax_rf.set_title(f"RF Permutation Importance (Top {top_n_fi})", fontsize=13)
                ax_rf.set_xlabel("Mean Importance ± Std")
                ax_rf.invert_yaxis()
                ax_rf.grid(axis="x", linestyle="--", alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig_rf); plt.close()

                # R² score
                from sklearn.metrics import r2_score
                y_pred_rf = rf.predict(X_fi)
                r2 = r2_score(y_fi, y_pred_rf)
                c1, c2 = st.columns(2)
                c1.metric("訓練集 R²", f"{r2:.3f}")
                c2.metric("特徵數", X_fi.shape[1])
                st.dataframe(perm_df.style.background_gradient(cmap="Blues", subset=["Perm_Importance"]),
                             width="stretch", hide_index=True)

            # ── SHAP ──
            with fi_subtabs[1]:
                st.markdown("#### SHAP 分析")
                st.info("SHAP 計算需要幾秒，點下方按鈕開始。")
                shap_subtabs = st.tabs(["Beeswarm", "Bar (全局)", "Waterfall (單筆)", "Dependence Plot"])

                if st.button("🔮 計算 SHAP Values", key="run_shap"):
                    try:
                        import shap as shap_lib
                        with st.spinner("計算 SHAP values..."):
                            explainer = shap_lib.TreeExplainer(rf)
                            shap_vals = explainer.shap_values(X_fi)
                        st.session_state["shap_vals"]     = shap_vals
                        st.session_state["shap_explainer"]= explainer
                        st.session_state["shap_lib"]      = shap_lib
                        st.success("✅ SHAP 計算完成！")
                    except ImportError:
                        st.error("請在 requirements.txt 加入 `shap` 並重新部署。")
                    except Exception as e:
                        st.error(f"SHAP 失敗：{e}")

                if st.session_state.get("shap_vals") is not None:
                    shap_vals  = st.session_state["shap_vals"]
                    shap_lib   = st.session_state["shap_lib"]
                    explainer  = st.session_state["shap_explainer"]

                    # helper: wrap long feature names for y-axis labels
                    def wrap_feat_names(names, width=35):
                        return ["\n".join(textwrap.wrap(n, width)) for n in names]

                    # helper: rename X_fi columns to short codes for SHAP plots, return mapping
                    def make_short_X(X):
                        short_names = {c: f"F{i:02d}" for i, c in enumerate(X.columns)}
                        return X.rename(columns=short_names), short_names

                    X_fi_short, short_map = make_short_X(X_fi)
                    reverse_map = {v: k for k, v in short_map.items()}
                    shap_vals_arr = np.array(shap_vals)  # ensure plain ndarray

                    with shap_subtabs[0]:
                        st.markdown("**Beeswarm Plot（特徵對預測的影響分布）**")
                        fig_height = max(6, top_n_fi * 0.55)
                        plt.figure(figsize=(11, fig_height))
                        shap_lib.summary_plot(shap_vals_arr, X_fi_short,
                                              max_display=top_n_fi,
                                              plot_type="dot", show=False)
                        # replace short codes with wrapped original names on y-axis
                        ax_bee = plt.gca()
                        ax_bee.set_yticklabels(
                            ["\n".join(textwrap.wrap(reverse_map.get(t.get_text(), t.get_text()), 40))
                             for t in ax_bee.get_yticklabels()], fontsize=8
                        )
                        plt.subplots_adjust(left=0.38)
                        st.pyplot(plt.gcf()); plt.close()

                    with shap_subtabs[1]:
                        st.markdown("**Global Bar Plot（平均絕對 SHAP）**")
                        fig_height = max(5, top_n_fi * 0.55)
                        plt.figure(figsize=(11, fig_height))
                        shap_lib.summary_plot(shap_vals_arr, X_fi_short,
                                              max_display=top_n_fi,
                                              plot_type="bar", show=False)
                        ax_bar_s = plt.gca()
                        ax_bar_s.set_yticklabels(
                            ["\n".join(textwrap.wrap(reverse_map.get(t.get_text(), t.get_text()), 40))
                             for t in ax_bar_s.get_yticklabels()], fontsize=8
                        )
                        plt.subplots_adjust(left=0.38)
                        st.pyplot(plt.gcf()); plt.close()

                    with shap_subtabs[2]:
                        st.markdown("**Waterfall Plot（單一樣本預測解釋）**")
                        sample_idx_shap = st.slider("選擇樣本編號", 0, len(X_fi)-1, 0, key="shap_sample")

                        # Fix: ensure base_values is a plain Python float (scalar)
                        ev_raw = explainer.expected_value
                        if hasattr(ev_raw, "__len__"):
                            base_val_scalar = float(ev_raw[0])
                        else:
                            base_val_scalar = float(ev_raw)

                        expl_obj = shap_lib.Explanation(
                            values=shap_vals_arr[sample_idx_shap],
                            base_values=base_val_scalar,          # ← must be scalar float
                            data=X_fi.iloc[sample_idx_shap].values,
                            feature_names=X_fi_short.columns.tolist()  # short codes first
                        )
                        fig_wf_h = max(6, top_n_fi * 0.55)
                        plt.figure(figsize=(12, fig_wf_h))
                        shap_lib.plots.waterfall(expl_obj, max_display=top_n_fi, show=False)
                        # replace short codes → wrapped original names
                        ax_wf = plt.gca()
                        ax_wf.set_yticklabels(
                            ["\n".join(textwrap.wrap(reverse_map.get(t.get_text(), t.get_text()), 40))
                             for t in ax_wf.get_yticklabels()], fontsize=8
                        )
                        plt.subplots_adjust(left=0.38)
                        st.pyplot(plt.gcf()); plt.close()

                        # show sample info
                        st.caption(f"樣本 {sample_idx_shap}｜預測值：{rf.predict(X_fi.iloc[[sample_idx_shap]])[0]:.3f}｜"
                                   f"實際值：{y_fi.iloc[sample_idx_shap]:.3f}｜基準值：{base_val_scalar:.3f}")

                    with shap_subtabs[3]:
                        st.markdown("**Dependence Plot（特定特徵的 SHAP 與交互效應）**")
                        dep_feat_orig    = st.selectbox("主特徵", X_fi.columns.tolist(), key="shap_dep_feat")
                        dep_interact_orig= st.selectbox("交互著色特徵（auto = 自動偵測）",
                                                    ["auto"] + X_fi.columns.tolist(), key="shap_dep_interact")
                        dep_feat_short    = short_map[dep_feat_orig]
                        interact_short    = None if dep_interact_orig == "auto" else short_map[dep_interact_orig]
                        fig_dep, ax_dep = plt.subplots(figsize=(9, 5))
                        shap_lib.dependence_plot(dep_feat_short, shap_vals_arr, X_fi_short,
                                                 interaction_index=interact_short, ax=ax_dep, show=False)
                        ax_dep.set_xlabel(dep_feat_orig, fontsize=9)
                        plt.tight_layout()
                        st.pyplot(fig_dep); plt.close()

            # ── PLS ──
            with fi_subtabs[2]:
                st.markdown("#### PLS — VIP Score")

                # CV curve for component selection
                st.markdown("**Step 1：先用交叉驗證決定最佳主成分數**")
                max_comp = min(10, X_fi.shape[1], len(y_fi) - 1)
                if max_comp < 2:
                    st.warning("樣本數或特徵數不足，無法執行 PLS CV。")
                else:
                    if st.button("📉 計算 PLS CV MSE", key="run_pls_cv"):
                        with st.spinner("PLS 交叉驗證中..."):
                            mse_list_pls = []
                            comp_range_pls = range(1, max_comp + 1)
                            for n_c in comp_range_pls:
                                pls_cv_model = PLSRegression(n_components=n_c)
                                # Fix: convert y to numpy to avoid DataFrame ambiguity
                                y_arr = np.array(y_fi).ravel()
                                y_cv_pls = cross_val_predict(pls_cv_model, X_fi.values, y_arr,
                                                             cv=min(5, len(y_arr)))
                                mse_list_pls.append(mean_squared_error(y_arr, y_cv_pls))
                            st.session_state["pls_mse"]  = mse_list_pls
                            st.session_state["pls_range"] = list(comp_range_pls)

                    if st.session_state.get("pls_mse") is not None:
                        mse_list_pls   = st.session_state["pls_mse"]
                        comp_range_pls = st.session_state["pls_range"]
                        best_n = comp_range_pls[int(np.argmin(mse_list_pls))]
                        fig_mse, ax_mse = plt.subplots(figsize=(8, 4))
                        ax_mse.plot(comp_range_pls, mse_list_pls, marker="o", color="#2e86ab")
                        ax_mse.axvline(best_n, color="#e84855", linestyle="--", label=f"Best = {best_n}")
                        ax_mse.set_xlabel("Number of Components")
                        ax_mse.set_ylabel("MSE (5-fold CV)")
                        ax_mse.set_title("PLS Cross-Validation MSE")
                        ax_mse.legend()
                        ax_mse.grid(linestyle="--", alpha=0.5)
                        plt.tight_layout()
                        st.pyplot(fig_mse); plt.close()
                        st.info(f"建議主成分數：**{best_n}**（MSE 最低）")

                    st.markdown("**Step 2：設定主成分數並計算 VIP**")
                    n_pls_comp = st.slider("PLS 主成分數", 1, max_comp, min(3, max_comp), key="pls_n_comp")

                    if st.button("📐 計算 PLS VIP", key="run_pls_vip"):
                        with st.spinner("PLS 訓練中..."):
                            pls_model = PLSRegression(n_components=n_pls_comp)
                            # Fix: use numpy arrays to avoid DataFrame ambiguity
                            X_arr = X_fi.values
                            y_arr = np.array(y_fi).ravel()
                            pls_model.fit(X_arr, y_arr)

                            # VIP calculation — fully scalar, sklearn-version safe
                            t = np.array(pls_model.x_scores_,  dtype=float)   # (n, h)
                            w = np.array(pls_model.x_weights_,  dtype=float)   # (p, h)
                            q = np.array(pls_model.y_loadings_, dtype=float).ravel()  # (h,)
                            p_feat, h = w.shape
                            # s[j]: scalar variance of component j weighted by y-loading²
                            s = np.array([
                                float(t[:, j] @ t[:, j]) * float(q[j]) ** 2
                                for j in range(h)
                            ], dtype=float)
                            total_s = float(s.sum())
                            # w_norm[j, i]: normalised weight of feature i for component j
                            w_col_norms = np.linalg.norm(w, axis=0)          # (h,)
                            w_normed = (w / w_col_norms) ** 2                 # (p, h)
                            # vip[i] = sqrt( p * sum_j(s[j] * w_normed[i,j]) / total_s )
                            vips = np.sqrt(p_feat * (w_normed @ s) / total_s)  # (p,)

                            vip_df_pls = pd.DataFrame({
                                "Feature": X_fi.columns,
                                "VIP": vips,
                            }).sort_values("VIP", ascending=False).reset_index(drop=True)
                            st.session_state["pls_vip_df"] = vip_df_pls

                    if st.session_state.get("pls_vip_df") is not None:
                        vip_df_pls = st.session_state["pls_vip_df"]
                        top_vip = vip_df_pls.head(top_n_fi)
                        fig_vip, ax_vip = plt.subplots(figsize=(10, max(5, top_n_fi * 0.45)))
                        vip_colors = ["#e84855" if v >= 1.0 else "#2e86ab" for v in top_vip["VIP"]]
                        ax_vip.barh(top_vip["Feature"], top_vip["VIP"], color=vip_colors, alpha=0.85)
                        ax_vip.axvline(1.0, color="#e84855", linestyle="--", lw=1.5, label="VIP=1 (重要門檻)")
                        ax_vip.set_title(f"PLS VIP Score (Top {top_n_fi})", fontsize=13)
                        ax_vip.set_xlabel("VIP Score")
                        ax_vip.invert_yaxis()
                        ax_vip.legend()
                        ax_vip.grid(axis="x", linestyle="--", alpha=0.5)
                        plt.tight_layout()
                        st.pyplot(fig_vip); plt.close()
                        st.caption("VIP ≥ 1.0 的特徵（紅色）通常被視為對目標變數有顯著影響。")
                        st.dataframe(vip_df_pls.style.background_gradient(cmap="Reds", subset=["VIP"]),
                                     width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────
# TAB 7: 文獻佐證分析（PubMed + Gemini）
# ─────────────────────────────────────────────────────────
with tabs[7]:
    st.header("📚 文獻佐證分析")
    st.markdown("""
    <div class="info-box">
    自動從 <b>PubMed</b> 搜尋真實論文 → 抓取摘要 → Gemini 基於真實文獻整理分析。
    所有引用都附 PMID 連結，可直接追蹤原始論文。
    </div>
    """, unsafe_allow_html=True)

    import urllib.parse as _uparse
    import urllib.request as _ureq2
    import json as _ujson2
    import time as _time2
    import re as _re2
    import os as _os2

    def pubmed_search(query, max_results=5):
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        url = base + "esearch.fcgi?db=pubmed&retmode=json&retmax=" + str(max_results) + "&term=" + _uparse.quote(query)
        try:
            with _ureq2.urlopen(url, timeout=10) as r:
                data = _ujson2.loads(r.read())
            return data["esearchresult"].get("idlist", [])
        except Exception:
            return []

    def pubmed_fetch_abstracts(pmids):
        if not pmids:
            return []
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        fetch_url = base + "efetch.fcgi?db=pubmed&id=" + ",".join(pmids) + "&rettype=abstract&retmode=xml"
        try:
            with _ureq2.urlopen(fetch_url, timeout=15) as r:
                xml = r.read().decode("utf-8")
        except Exception:
            return []

        articles = []
        blocks = _re2.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, _re2.DOTALL)
        for block in blocks:
            pmid_m    = _re2.search(r"<PMID[^>]*>(\d+)</PMID>", block)
            title_m   = _re2.search(r"<ArticleTitle>(.*?)</ArticleTitle>", block, _re2.DOTALL)
            abs_m     = _re2.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", block, _re2.DOTALL)
            year_m    = _re2.search(r"<PubDate>.*?<Year>(\d{4})</Year>", block, _re2.DOTALL)
            journal_m = _re2.search(r"<ISOAbbreviation>(.*?)</ISOAbbreviation>", block)

            def clean_xml(s):
                return _re2.sub(r"<[^>]+>", "", s).strip() if s else ""

            pmid    = pmid_m.group(1)    if pmid_m    else "?"
            title   = clean_xml(title_m.group(1))   if title_m   else "No title"
            abstract= clean_xml(abs_m.group(1))      if abs_m     else "No abstract"
            year    = year_m.group(1)    if year_m    else "?"
            journal = clean_xml(journal_m.group(1))  if journal_m else "?"

            articles.append({
                "pmid": pmid, "title": title,
                "abstract": abstract[:800],
                "year": year, "journal": journal,
                "url": "https://pubmed.ncbi.nlm.nih.gov/" + pmid + "/"
            })
        return articles

    def build_search_queries(features, target, context):
        proc_kw = context.split(",")[0].strip() if context else "bioprocess"
        queries = []
        for feat in features:
            clean = _re2.sub(r"\(.*?\)", "", feat).replace("_", " ").replace("-", " ")
            words = [w for w in clean.split() if len(w) > 3]
            kw = " ".join(words[:4])
            if kw:
                queries.append((feat, kw + " " + proc_kw + " chromatography yield"))
        queries.append(("Overall", proc_kw + " yield optimization process parameters"))
        return queries

    def call_gemini_lit(api_key, prompt, max_tokens=6000):
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "gemini-2.5-flash:generateContent?key=" + api_key)
        payload = _ujson2.dumps({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2},
        }).encode("utf-8")
        req = _ureq2.Request(url, data=payload,
                             headers={"Content-Type": "application/json"}, method="POST")
        with _ureq2.urlopen(req, timeout=60) as resp:
            result = _ujson2.loads(resp.read())
        return result["candidates"][0]["content"]["parts"][0]["text"]

    # ── Step 1：參數設定 ──────────────────────────────────────
    st.markdown("### Step 1：設定分析參數")
    col_la, col_lb = st.columns(2)
    with col_la:
        target_var_lit = st.text_input(
            "🎯 目標變數（Y）",
            value=st.session_state.get("target_col", ""),
            placeholder="例：phenyl chromatography_Yield Rate (%)",
        )
        process_context = st.text_input(
            "🧪 製程背景",
            value="rhG-CSF protein purification, Phenyl Hydrophobic Interaction Chromatography",
        )
        max_papers_per_feat = st.slider("每個參數搜尋論文數", 1, 5, 3, key="lit_max_papers")

    with col_lb:
        auto_features = []
        if st.session_state.get("fi_perm_df") is not None:
            auto_features = st.session_state["fi_perm_df"]["Feature"].head(8).tolist()
        if st.session_state.get("pls_vip_df") is not None:
            vip_top = st.session_state["pls_vip_df"][
                st.session_state["pls_vip_df"]["VIP"] >= 1.0]["Feature"].head(8).tolist()
            auto_features = list(dict.fromkeys(auto_features + vip_top))

        important_features_text = st.text_area(
            "📌 重要參數（每行一個）",
            value="\n".join(auto_features[:6]) if auto_features else "",
            height=200,
        )
        lang_lit = st.radio("輸出語言", ["繁體中文", "English"], horizontal=True, key="lit_lang")

    important_features = [f.strip() for f in important_features_text.strip().split("\n") if f.strip()]

    # ── Step 2：PubMed 搜尋 ───────────────────────────────────
    st.markdown("### Step 2：搜尋 PubMed 文獻")

    if not important_features or not target_var_lit.strip():
        st.warning("請填入目標變數與重要參數。")
    else:
        if st.button("🔎 搜尋 PubMed 論文", key="run_pubmed"):
            queries = build_search_queries(important_features, target_var_lit, process_context)
            all_articles = {}
            prog = st.progress(0, text="搜尋中...")
            for i, (feat, query) in enumerate(queries):
                pmids = pubmed_search(query, max_results=max_papers_per_feat)
                if pmids:
                    arts = pubmed_fetch_abstracts(pmids)
                    all_articles[feat] = {"query": query, "articles": arts}
                _time2.sleep(0.35)
                prog.progress((i + 1) / len(queries), text="搜尋中：" + feat[:40] + "...")
            prog.empty()
            st.session_state["pubmed_results"] = all_articles
            total = sum(len(v["articles"]) for v in all_articles.values())
            st.success("✅ 共找到 " + str(total) + " 篇論文！")

        if st.session_state.get("pubmed_results"):
            all_articles = st.session_state["pubmed_results"]
            st.markdown("#### 📄 找到的論文")
            for feat, data in all_articles.items():
                with st.expander("**" + feat + "** — " + str(len(data["articles"])) + " 篇 | 搜尋：`" + data["query"] + "`"):
                    for art in data["articles"]:
                        st.markdown(
                            "**[" + art["title"] + "](" + art["url"] + ")**  \n"
                            "_" + art["journal"] + "_ (" + art["year"] + ") · PMID: `" + art["pmid"] + "`  \n"
                            + art["abstract"] + "..."
                        )
                        st.markdown("---")

            # ── Step 3：Gemini 分析 ──────────────────────────────
            st.markdown("### Step 3：AI 基於真實文獻分析")

            if st.button("🧠 開始 AI 文獻分析", type="primary", key="run_lit_gemini"):
                _api_key_lit = st.secrets.get("GEMINI_API_KEY", _os2.environ.get("GEMINI_API_KEY", ""))
                if not _api_key_lit:
                    st.error("找不到 GEMINI_API_KEY，請在 Streamlit Secrets 設定。")
                    st.stop()

                lit_context = ""
                ref_list = []
                ref_idx = 1
                for feat, data in all_articles.items():
                    lit_context += "\n\n=== Parameter: " + feat + " ===\n"
                    for art in data["articles"]:
                        lit_context += (
                            "[" + str(ref_idx) + "] " + art["title"] + " ("
                            + art["journal"] + ", " + art["year"] + ", PMID:" + art["pmid"] + ")\n"
                            "Abstract: " + art["abstract"] + "\n\n"
                        )
                        ref_list.append(
                            "[" + str(ref_idx) + "] " + art["title"] + ". "
                            + art["journal"] + " (" + art["year"] + "). "
                            + "PMID: " + art["pmid"] + ". " + art["url"]
                        )
                        ref_idx += 1

                lang_inst = "Respond in Traditional Chinese (繁體中文)." if lang_lit == "繁體中文" else "Respond in English."
                feat_list = "\n".join(["  " + str(i+1) + ". " + f for i, f in enumerate(important_features)])

                prompt = (
                    "You are an expert bioprocess scientist. Analyze the following real PubMed literature "
                    "to explain why the listed process parameters are important predictors of the target variable.\n\n"
                    "Process Context: " + process_context + "\n"
                    "Target Variable: " + target_var_lit + "\n\n"
                    "Important Parameters:\n" + feat_list + "\n\n"
                    "=== REAL PUBMED LITERATURE ===\n" + lit_context + "\n\n"
                    "STRICT RULES:\n"
                    "1. ONLY cite papers from the list above using [number]\n"
                    "2. Do NOT invent papers\n"
                    "3. If evidence is weak or absent, say so\n"
                    "4. End with a Reference List with PMID and URL\n"
                    "5. " + lang_inst + "\n\n"
                    "Output:\n"
                    "## 總覽\n(collective effect on target)\n\n"
                    "## 各參數分析\n(each param: mechanism + [ref])\n\n"
                    "## 研究缺口\n(what is NOT covered by found literature)\n\n"
                    "## 參考文獻\n(all cited with PMID + URL)"
                )

                with st.spinner("🧠 Gemini 正在基於真實文獻分析（約 30-60 秒）..."):
                    try:
                        ai_response = call_gemini_lit(_api_key_lit, prompt)
                        st.session_state["lit_response"] = ai_response
                        st.session_state["lit_ref_list"] = ref_list
                        st.session_state["lit_params"] = {
                            "target": target_var_lit,
                            "features": important_features,
                            "context": process_context,
                        }
                        st.success("✅ 分析完成！")
                    except Exception as e:
                        import traceback
                        st.error("Gemini 呼叫失敗：" + str(e))
                        if hasattr(e, "read"):
                            try:
                                st.code(e.read().decode("utf-8"), language="json")
                            except Exception:
                                pass
                        st.code(traceback.format_exc())

    # ── 顯示分析結果 ──────────────────────────────────────────
    if st.session_state.get("lit_response"):
        params   = st.session_state.get("lit_params", {})
        ref_list = st.session_state.get("lit_ref_list", [])

        st.markdown("---")
        st.markdown("### 📄 分析結果")
        st.caption(
            "目標：`" + params.get("target", "") + "` ｜ "
            "參數數：" + str(len(params.get("features", []))) + " ｜ "
            "引用論文：" + str(len(ref_list)) + " 篇"
        )
        st.markdown(st.session_state["lit_response"])

        if ref_list:
            st.markdown("### 🔗 快速論文連結")
            for ref in ref_list:
                url_m = _re2.search(r"https://pubmed[^\s]+", ref)
                if url_m:
                    label = ref.split(url_m.group())[0].rstrip(". ")
                    st.markdown("- " + label + " [🔗 PubMed](" + url_m.group() + ")")

        st.markdown("---")
        export_md = (
            "# 文獻佐證分析報告\n"
            "生成時間：" + _time2.strftime("%Y-%m-%d %H:%M") + "\n\n"
            "## 製程背景\n" + params.get("context", "") + "\n\n"
            "## 目標變數\n" + params.get("target", "") + "\n\n"
            "## 分析參數\n" + "\n".join(["- " + f for f in params.get("features", [])]) + "\n\n"
            "---\n\n"
            "## AI 分析結果（基於 PubMed 真實文獻）\n\n"
            + st.session_state["lit_response"] + "\n\n"
            "---\n\n"
            "## 所有搜尋到的論文\n\n"
            + "\n".join(ref_list)
        )
        st.download_button(
            "📥 下載完整報告（含文獻出處）",
            data=export_md.encode("utf-8"),
            file_name="pubmed_literature_report.md",
            mime="text/markdown",
            key="download_lit"
        )

        st.markdown("### 💬 追問")
        follow_up = st.text_area(
            "針對以上分析，想進一步了解？",
            placeholder="例：Loading capacity 對 yield 的非線性效應，文獻中有哪些實驗數據？",
            key="lit_followup_q"
        )
        if st.button("📨 送出追問", key="lit_followup_btn"):
            if follow_up.strip():
                _api_key_fu = st.secrets.get("GEMINI_API_KEY", _os2.environ.get("GEMINI_API_KEY", ""))
                with st.spinner("思考中..."):
                    try:
                        lang_fu = "Respond in Traditional Chinese." if lang_lit == "繁體中文" else "Respond in English."
                        fu_prompt = (
                            "Based on this previous analysis:\n\n"
                            + st.session_state["lit_response"]
                            + "\n\nUser follow-up: " + follow_up
                            + "\n\nAnswer using ONLY the cited literature. "
                            "If more papers are needed, say so clearly.\n" + lang_fu
                        )
                        reply = call_gemini_lit(_api_key_fu, fu_prompt, max_tokens=2000)
                        st.markdown("#### 💬 回覆")
                        st.markdown(reply)
                    except Exception as e:
                        st.error("追問失敗：" + str(e))

