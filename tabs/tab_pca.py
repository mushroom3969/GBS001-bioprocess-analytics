"""Tab 5 — PCA 主成分分析"""
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import f as f_dist
from sklearn.preprocessing import StandardScaler
from pca import pca

from utils import extract_number, plot_ht2_bar, plot_contribution_bar


def _ht2_threshold(alpha: float, n: int, p: int) -> float:
    f_crit = f_dist.ppf(1 - alpha, p, n - p)
    return (p * (n - 1) * (n + 1)) / (n * (n - p)) * f_crit


def render(selected_process_df):
    st.header("PCA 主成分分析")
    _cd = st.session_state.get("clean_df")
    work_df = _cd if _cd is not None else selected_process_df

    if work_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    numeric_options = work_df.select_dtypes(include=["number"]).columns.tolist()
    stored_target   = st.session_state.get("target_col")
    default_idx     = numeric_options.index(stored_target) if stored_target in numeric_options else 0

    with st.expander("⚙️ PCA 設定", expanded=True):
        pc1, pc2, pc3 = st.columns(3)
        target_col_pca = pc1.selectbox("排除的目標欄位（Y）", numeric_options,
                                        index=default_idx, key="pca_target")
        n_components   = pc2.slider("最大主成分數", 2, min(15, len(numeric_options) - 1), 5)
        alpha_pca      = pc3.select_slider("Hotelling T² 顯著水準 α",
                                            [0.01, 0.05, 0.10], value=0.05)

    if st.button("🧩 執行 PCA", key="run_pca"):
        try:
            exclude = [c for c in ["BatchID", target_col_pca] if c in work_df.columns]
            X_pca   = work_df.drop(columns=exclude, errors="ignore")
            X_pca   = X_pca.select_dtypes(include=["number"]).dropna(axis=1)

            if X_pca.shape[1] < 2:
                st.error("有效欄位不足 2 個，無法執行 PCA。")
                return

            label_pca    = work_df["BatchID"].values if "BatchID" in work_df.columns else np.arange(len(work_df))
            scaler       = StandardScaler()
            x_scaled     = scaler.fit_transform(X_pca)
            feat_names   = X_pca.columns.tolist()
            n_comp_actual = min(n_components, X_pca.shape[1] - 1)

            with st.spinner("PCA 計算中..."):
                model = pca(n_components=n_comp_actual, alpha=alpha_pca,
                            detect_outliers=["ht2", "spe"])
                results = model.fit_transform(x_scaled)

            st.session_state.update({
                "pca_model": model, "pca_results": results,
                "pca_X": X_pca, "pca_x_scaled": x_scaled,
                "pca_labels": label_pca, "pca_feat": feat_names,
            })
            st.success("✅ PCA 完成！")
        except Exception as e:
            st.error(f"PCA 執行失敗：{e}")

    if st.session_state.get("pca_model") is None:
        return

    # ── 讀取 session ──────────────────────────────────────────
    model        = st.session_state["pca_model"]
    results      = st.session_state["pca_results"]
    X_pca        = st.session_state["pca_X"]
    x_scaled     = st.session_state["pca_x_scaled"]
    label_pca    = st.session_state["pca_labels"]
    feat_names   = st.session_state["pca_feat"]

    ev       = model.results["explained_var"]
    vr       = model.results["variance_ratio"]
    scores   = results["PC"].values
    loadings = model.results["loadings"].values
    n_pc     = scores.shape[1]

    # ── Metrics ───────────────────────────────────────────────
    cols_m = st.columns(min(n_pc, 5))
    for i, c in enumerate(cols_m):
        c.metric(f"PC{i+1} 累計解釋", f"{ev[i]*100:.1f}%", delta=f"+{vr[i]*100:.1f}%")

    pca_subtabs = st.tabs([
        "📊 Scree & Scatter", "🔵 Biplot",
        "🚨 Hotelling T² 異常偵測", "🔬 單筆貢獻分析",
    ])

    # ── Subtab 0 ─────────────────────────────────────────────
    with pca_subtabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Scree Plot**")
            fig_sc, _ = model.plot()
            st.pyplot(fig_sc); plt.close()
        with c2:
            st.markdown("**Score Scatter (PC1 vs PC2)**")
            fig_s2, _ = model.scatter(SPE=True, HT2=True)
            st.pyplot(fig_s2); plt.close()

    # ── Subtab 1 — Biplot ─────────────────────────────────────
    with pca_subtabs[1]:
        bp1, bp2 = st.columns(2)
        n_feat_bi = bp1.slider("顯示載荷向量數", 3, min(15, len(feat_names)), 6, key="bi_nfeat")
        pc_opts   = list(range(n_pc))
        pc_x = bp2.selectbox("X 軸 PC", pc_opts, index=0,
                              format_func=lambda x: f"PC{x+1}", key="bi_pcx")
        pc_y = bp2.selectbox("Y 軸 PC", pc_opts, index=min(1, n_pc - 1),
                              format_func=lambda x: f"PC{x+1}", key="bi_pcy")
        fig_bi, _ = model.biplot(n_feat=n_feat_bi, PC=[pc_x, pc_y],
                                  legend=True, SPE=True, HT2=True)
        st.pyplot(fig_bi); plt.close()

        st.markdown("**Top Features per PC**")
        topfeat = model.results.get("topfeat")
        if topfeat is not None:
            tf = topfeat.copy()
            def safe_map(x):
                try:
                    idx = int(x)
                    return feat_names[idx] if 0 <= idx < len(feat_names) else str(x)
                except Exception:
                    return str(x)
            tf["feature_name"] = tf["feature"].apply(safe_map)
            st.dataframe(tf, width="stretch", hide_index=True)

    # ── Subtab 2 — Hotelling T² ───────────────────────────────
    with pca_subtabs[2]:
        st.markdown("#### Hotelling T² — 各 Batch 異常程度")
        ht2_vals = np.sum((scores ** 2) / ev, axis=1)
        n_obs, p = x_scaled.shape[0], n_pc
        thres_68 = _ht2_threshold(0.32, n_obs, p)
        thres_95 = _ht2_threshold(0.05, n_obs, p)
        thres_99 = _ht2_threshold(0.01, n_obs, p)

        def _classify(v):
            if v > thres_99: return "🔴 >99%"
            if v > thres_95: return "🟠 >95%"
            if v > thres_68: return "🟡 >68%"
            return "🟢 Normal"

        ht2_df = pd.DataFrame({"Batch": label_pca, "T²": ht2_vals.round(3)})
        ht2_df["Status"] = ht2_df["T²"].apply(_classify)
        ht2_df = ht2_df.sort_values("T²", ascending=False).reset_index(drop=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("超過 68% 閾值", f"{(ht2_vals > thres_68).sum()} 批")
        m2.metric("超過 95% 閾值", f"{(ht2_vals > thres_95).sum()} 批")
        m3.metric("超過 99% 閾值", f"{(ht2_vals > thres_99).sum()} 批")

        fig_ht = plot_ht2_bar(label_pca, ht2_vals, thres_68, thres_95, thres_99)
        st.pyplot(fig_ht); plt.close()
        st.markdown("**Batch 排行（T² 由大到小）**")
        st.dataframe(ht2_df, width="stretch", hide_index=True)

    # ── Subtab 3 — 單筆貢獻 ───────────────────────────────────
    with pca_subtabs[3]:
        st.markdown("#### 選擇要分析的 Batch")
        batch_options = [str(b) for b in label_pca]
        sel_batch = st.selectbox("選擇 Batch", batch_options, key="pca_sel_batch")
        sample_i  = batch_options.index(sel_batch)

        view_mode = st.radio(
            "分析模式",
            ["所有 PC 的特徵貢獻（總 T²）", "單一 PC 的特徵貢獻"],
            horizontal=True, key="pca_view_mode",
        )
        top_n_contrib = st.slider("顯示前 N 個特徵", 5, min(30, len(feat_names)), 15, key="pca_top_contrib")

        if view_mode == "所有 PC 的特徵貢獻（總 T²）":
            contributions = np.zeros(loadings.shape[1])
            for a in range(n_pc):
                contributions += (scores[sample_i, a] / ev[a]) * loadings[a, :] * x_scaled[sample_i, :]

            df_contrib = (
                pd.DataFrame({"Feature": feat_names, "Contribution": contributions})
                .reindex(pd.Series(contributions).abs().sort_values(ascending=False).index)
                .reset_index(drop=True)
                .head(top_n_contrib)
            )
            fig_c = plot_contribution_bar(df_contrib, f"Hotelling T² Total Contribution — {sel_batch}")
            st.pyplot(fig_c); plt.close()
            st.dataframe(df_contrib, width="stretch", hide_index=True)

            # PC-wise decomposition
            st.markdown("**各 PC 對該 Batch T² 的貢獻分解**")
            t2_per_pc = (scores[sample_i, :] ** 2) / ev
            fig_pc_bar, ax_pc = plt.subplots(figsize=(8, 4))
            pc_labels_bar = [f"PC{j+1}" for j in range(n_pc)]
            ax_pc.bar(pc_labels_bar, t2_per_pc, color="#2e86ab", alpha=0.8)
            for bar, val in zip(ax_pc.patches, t2_per_pc):
                ax_pc.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           f"{val:.2f}", ha="center", va="bottom", fontsize=9)
            ax_pc.set_ylabel("T² per PC")
            ax_pc.set_title(f"PC-wise T² Decomposition — {sel_batch}")
            ax_pc.grid(axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_pc_bar); plt.close()

        else:
            sel_pc   = st.selectbox("選擇 PC", list(range(n_pc)),
                                     format_func=lambda x: f"PC{x+1}", key="pca_sel_pc")
            pc_contrib = (scores[sample_i, sel_pc] / ev[sel_pc]) * loadings[sel_pc, :] * x_scaled[sample_i, :]
            df_pc = (
                pd.DataFrame({"Feature": feat_names, "Contribution": pc_contrib})
                .reindex(pd.Series(pc_contrib).abs().sort_values(ascending=False).index)
                .reset_index(drop=True)
                .head(top_n_contrib)
            )
            fig_pc = plot_contribution_bar(df_pc, f"PC{sel_pc+1} Feature Contribution — {sel_batch}")
            st.pyplot(fig_pc); plt.close()
            st.dataframe(df_pc, width="stretch", hide_index=True)
