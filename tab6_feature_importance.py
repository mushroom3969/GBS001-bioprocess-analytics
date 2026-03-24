"""Tab 6: Feature Importance — RF, SHAP, PLS-VIP"""
import sys, os as _os
_dir = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_dir)
for _p in [_dir, _root, _os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

from utils import (
    train_rf_and_importance,
    make_short_feature_map, get_shap_base_value, restore_shap_yticklabels,
    compute_pls_vip, compute_pls_cv_mse,
)


def render(work_df):
    st.header("特徵重要性分析")
    if work_df is None:
        st.info("請先在側欄選擇製程步驟。")
        return

    numeric_cols     = work_df.select_dtypes(include=["number"]).columns.tolist()
    stored_target    = st.session_state.get("target_col")
    default_fi       = numeric_cols.index(stored_target) if stored_target in numeric_cols else 0
    target_col_fi    = st.selectbox("目標欄位（Y）", numeric_cols, index=default_fi, key="fi_target")
    top_n_fi         = st.slider("顯示前 N 個特徵", 5, 30, 15, key="fi_topn")

    if st.button("🚀 訓練 Random Forest 模型", key="run_fi_prepare"):
        try:
            exclude = [c for c in ["BatchID", target_col_fi] if c in work_df.columns]
            X_fi    = work_df.drop(columns=exclude, errors="ignore").select_dtypes(include=["number"])
            y_fi    = work_df[target_col_fi]
            valid   = y_fi.notna() & X_fi.notna().all(axis=1)
            X_fi    = X_fi[valid].reset_index(drop=True)
            y_fi    = y_fi[valid].reset_index(drop=True)

            with st.spinner("Random Forest 訓練中..."):
                perm_df, rf, r2 = train_rf_and_importance(X_fi, y_fi)

            st.session_state.update({
                "fi_X": X_fi, "fi_y": y_fi,
                "fi_rf": rf,  "fi_perm_df": perm_df, "fi_r2": r2,
                "fi_target_col": target_col_fi,
                # Clear stale SHAP / PLS results when retraining
                "shap_vals": None, "pls_vip_df": None, "pls_mse": None,
            })
            st.success(f"✅ 訓練完成！訓練集 R² = {r2:.3f}")
        except Exception as e:
            st.error(f"訓練失敗：{e}")

    if st.session_state.get("fi_rf") is None:
        return

    X_fi    = st.session_state["fi_X"]
    y_fi    = st.session_state["fi_y"]
    rf      = st.session_state["fi_rf"]
    perm_df = st.session_state["fi_perm_df"]
    r2      = st.session_state.get("fi_r2", 0)

    subtabs = st.tabs(["🌲 RF 重要性", "🔮 SHAP 分析", "📐 PLS-VIP"])

    # ── RF ───────────────────────────────────────────────────
    with subtabs[0]:
        st.markdown("#### Random Forest — Permutation Importance")
        top_perm = perm_df.head(top_n_fi)
        fig, ax  = plt.subplots(figsize=(10, max(5, top_n_fi * 0.45)))
        ax.barh(top_perm["Feature"], top_perm["Perm_Importance"],
                xerr=top_perm["Std"],
                color=["#2e86ab" if v >= 0 else "#e84855" for v in top_perm["Perm_Importance"]],
                alpha=0.85, error_kw={"ecolor": "gray", "capsize": 3})
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"RF Permutation Importance (Top {top_n_fi})", fontsize=13)
        ax.set_xlabel("Mean Importance ± Std")
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        c1, c2 = st.columns(2)
        c1.metric("訓練集 R²", f"{r2:.3f}")
        c2.metric("特徵數", X_fi.shape[1])
        st.dataframe(perm_df.style.background_gradient(cmap="Blues", subset=["Perm_Importance"]),
                     width="stretch", hide_index=True)

    # ── SHAP ─────────────────────────────────────────────────
    with subtabs[1]:
        st.markdown("#### SHAP 分析")
        shap_subtabs = st.tabs(["Beeswarm", "Bar (全局)", "Waterfall (單筆)", "Dependence Plot"])

        if st.button("🔮 計算 SHAP Values", key="run_shap"):
            try:
                import shap as shap_lib
                with st.spinner("計算 SHAP values..."):
                    explainer = shap_lib.TreeExplainer(rf)
                    shap_vals = explainer.shap_values(X_fi)
                st.session_state.update({
                    "shap_vals": shap_vals,
                    "shap_explainer": explainer,
                    "shap_lib": shap_lib,
                })
                st.success("✅ SHAP 計算完成！")
            except ImportError:
                st.error("請在 requirements.txt 加入 `shap`。")
            except Exception as e:
                st.error(f"SHAP 失敗：{e}")

        if st.session_state.get("shap_vals") is not None:
            shap_vals = st.session_state["shap_vals"]
            explainer = st.session_state["shap_explainer"]
            shap_lib  = st.session_state["shap_lib"]

            X_fi_short, short_map, reverse_map = make_short_feature_map(X_fi)
            shap_arr = np.array(shap_vals)

            def fix_labels(ax):
                restore_shap_yticklabels(ax, reverse_map)
                plt.subplots_adjust(left=0.38)

            with shap_subtabs[0]:
                st.markdown("**Beeswarm Plot**")
                plt.figure(figsize=(11, max(6, top_n_fi * 0.55)))
                shap_lib.summary_plot(shap_arr, X_fi_short, max_display=top_n_fi,
                                      plot_type="dot", show=False)
                fix_labels(plt.gca())
                st.pyplot(plt.gcf()); plt.close()

            with shap_subtabs[1]:
                st.markdown("**Global Bar Plot**")
                plt.figure(figsize=(11, max(5, top_n_fi * 0.55)))
                shap_lib.summary_plot(shap_arr, X_fi_short, max_display=top_n_fi,
                                      plot_type="bar", show=False)
                fix_labels(plt.gca())
                st.pyplot(plt.gcf()); plt.close()

            with shap_subtabs[2]:
                st.markdown("**Waterfall Plot（單一樣本）**")
                idx = st.slider("選擇樣本編號", 0, len(X_fi)-1, 0, key="shap_sample")
                base_val = get_shap_base_value(explainer)
                expl_obj = shap_lib.Explanation(
                    values=shap_arr[idx],
                    base_values=base_val,
                    data=X_fi.iloc[idx].values,
                    feature_names=X_fi_short.columns.tolist(),
                )
                plt.figure(figsize=(12, max(6, top_n_fi * 0.55)))
                shap_lib.plots.waterfall(expl_obj, max_display=top_n_fi, show=False)
                fix_labels(plt.gca())
                st.pyplot(plt.gcf()); plt.close()
                st.caption(f"樣本 {idx} ｜預測：{rf.predict(X_fi.iloc[[idx]])[0]:.3f}"
                           f"｜實際：{y_fi.iloc[idx]:.3f}｜基準：{base_val:.3f}")

            with shap_subtabs[3]:
                st.markdown("**Dependence Plot**")
                dep_feat     = st.selectbox("主特徵", X_fi.columns.tolist(), key="shap_dep_feat")
                dep_interact = st.selectbox("交互著色特徵",
                                            ["auto"] + X_fi.columns.tolist(), key="shap_dep_interact")
                interact_short = None if dep_interact == "auto" else short_map[dep_interact]
                fig, ax = plt.subplots(figsize=(9, 5))
                shap_lib.dependence_plot(short_map[dep_feat], shap_arr, X_fi_short,
                                         interaction_index=interact_short, ax=ax, show=False)
                ax.set_xlabel(dep_feat, fontsize=9)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

    # ── PLS-VIP ──────────────────────────────────────────────
    with subtabs[2]:
        st.markdown("#### PLS — VIP Score")
        max_comp = min(10, X_fi.shape[1], len(y_fi) - 1)
        if max_comp < 2:
            st.warning("樣本數或特徵數不足。")
            return

        st.markdown("**Step 1：交叉驗證選擇最佳主成分數**")
        if st.button("📉 計算 PLS CV MSE", key="run_pls_cv"):
            with st.spinner("PLS 交叉驗證中..."):
                mse_list = compute_pls_cv_mse(X_fi, y_fi, max_components=max_comp)
                st.session_state["pls_mse"]   = mse_list
                st.session_state["pls_range"] = list(range(1, max_comp + 1))

        if st.session_state.get("pls_mse") is not None:
            mse_list   = st.session_state["pls_mse"]
            comp_range = st.session_state["pls_range"]
            best_n     = comp_range[int(np.argmin(mse_list))]
            fig, ax    = plt.subplots(figsize=(8, 4))
            ax.plot(comp_range, mse_list, marker="o", color="#2e86ab")
            ax.axvline(best_n, color="#e84855", linestyle="--", label=f"Best = {best_n}")
            ax.set(xlabel="Number of Components", ylabel="MSE (5-fold CV)",
                   title="PLS Cross-Validation MSE")
            ax.legend(); ax.grid(linestyle="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.info(f"建議主成分數：**{best_n}**")

        st.markdown("**Step 2：計算 VIP**")
        n_pls = st.slider("PLS 主成分數", 1, max_comp, min(3, max_comp), key="pls_n_comp")
        if st.button("📐 計算 PLS VIP", key="run_pls_vip"):
            with st.spinner("PLS 訓練中..."):
                vip_df, _ = compute_pls_vip(X_fi, y_fi, n_components=n_pls)
                st.session_state["pls_vip_df"] = vip_df

        if st.session_state.get("pls_vip_df") is not None:
            vip_df  = st.session_state["pls_vip_df"]
            top_vip = vip_df.head(top_n_fi)
            fig, ax = plt.subplots(figsize=(10, max(5, top_n_fi * 0.45)))
            ax.barh(top_vip["Feature"], top_vip["VIP"],
                    color=["#e84855" if v >= 1.0 else "#2e86ab" for v in top_vip["VIP"]],
                    alpha=0.85)
            ax.axvline(1.0, color="#e84855", linestyle="--", lw=1.5, label="VIP=1 (重要門檻)")
            ax.set_title(f"PLS VIP Score (Top {top_n_fi})", fontsize=13)
            ax.set_xlabel("VIP Score")
            ax.invert_yaxis(); ax.legend()
            ax.grid(axis="x", linestyle="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.caption("VIP ≥ 1.0 的特徵（紅色）對目標變數有顯著影響。")
            st.dataframe(vip_df.style.background_gradient(cmap="Reds", subset=["VIP"]),
                         width="stretch", hide_index=True)
