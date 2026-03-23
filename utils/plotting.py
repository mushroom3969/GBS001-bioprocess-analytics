"""
繪圖工具函式
"""

import math
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from .data_processing import extract_number, extract_batch_logic


def _setup_subplots(
    n_plots: int,
    cols_per_row: int,
    subplot_w: float = 5.0,
    subplot_h: float = 4.0,
) -> tuple:
    """建立統一規格的子圖網格，回傳 (fig, axes_flat)。"""
    num_rows = math.ceil(n_plots / cols_per_row)
    fig, axes = plt.subplots(
        num_rows, cols_per_row,
        figsize=(cols_per_row * subplot_w, num_rows * subplot_h),
    )
    axes_flat = np.array(axes).flatten()
    # 關閉多餘子圖
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].axis("off")
    return fig, axes_flat


def plot_indexed_lineplots(
    df: pd.DataFrame,
    batch_col: str = "BatchID",
    cols_per_row: int = 3,
) -> plt.Figure | None:
    """以批次序號為 X 軸繪製所有數值欄位的折線圖。"""
    sorted_df = df.copy()
    sorted_df["_sort"] = sorted_df[batch_col].apply(extract_number)
    sorted_df = sorted_df.sort_values("_sort").reset_index(drop=True)
    sorted_df["Sequence_Index"] = sorted_df.index + 1

    exclude = {batch_col, "_sort", "Sequence_Index"}
    numeric_cols = [
        c for c in sorted_df.select_dtypes(include=["number"]).columns
        if c not in exclude
    ]
    if not numeric_cols:
        st.warning("沒有找到可繪圖的數值型欄位。")
        return None

    sns.set_style("whitegrid")
    fig, axes = _setup_subplots(len(numeric_cols), cols_per_row)

    for i, col in enumerate(numeric_cols):
        sns.lineplot(
            data=sorted_df, x="Sequence_Index", y=col,
            marker="o", color="royalblue", linewidth=1.5, ax=axes[i],
        )
        axes[i].set_title("\n".join(textwrap.wrap(col, width=30)), fontsize=9, pad=8)
        axes[i].set_xlabel("Batch Sequence")
        axes[i].set_ylabel("Value")

    plt.tight_layout()
    return fig


def plot_clean_lineplots(
    df: pd.DataFrame,
    batch_col: str = "BatchID",
    cols_per_row: int = 3,
) -> plt.Figure | None:
    """以批次 YYNN 數字為 X 軸繪製趨勢圖。"""
    temp_df = df.copy()
    temp_df["_sort"] = temp_df[batch_col].apply(extract_batch_logic)
    temp_df = temp_df.sort_values("_sort")

    numeric_cols = [
        c for c in temp_df.select_dtypes(include=["number"]).columns
        if c not in {"_sort", "batch_num"}
    ]
    if not numeric_cols:
        return None

    sns.set_style("whitegrid")
    fig, axes = _setup_subplots(len(numeric_cols), cols_per_row)

    for i, col in enumerate(numeric_cols):
        sns.lineplot(data=temp_df, x="_sort", y=col, marker="o", color="teal", ax=axes[i])
        axes[i].xaxis.set_major_formatter(plt.FormatStrFormatter("%d"))
        axes[i].set_title("\n".join(textwrap.wrap(col, width=30)), fontsize=9, pad=8)
        axes[i].set_xlabel("Batch (YYNN)")
        axes[i].set_ylabel("Value")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    return fig


def plot_ht2_bar(
    label_pca: np.ndarray,
    ht2_vals: np.ndarray,
    thres_68: float,
    thres_95: float,
    thres_99: float,
) -> plt.Figure:
    """繪製 Hotelling T² 條狀圖，三段閾值線以不同顏色標示。"""
    idx_sorted = np.argsort([extract_number(str(b)) for b in label_pca])
    x_plot = np.arange(len(label_pca))

    bar_colors = [
        "#e84855" if v > thres_99
        else "#f4a261" if v > thres_95
        else "#e9c46a" if v > thres_68
        else "#2e86ab"
        for v in ht2_vals[idx_sorted]
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x_plot, ht2_vals[idx_sorted], color=bar_colors, alpha=0.85, width=0.7)
    ax.axhline(thres_68, color="#e9c46a", linestyle="--", lw=1.5, label=f"68% ({thres_68:.1f})")
    ax.axhline(thres_95, color="#f4a261", linestyle="--", lw=1.5, label=f"95% ({thres_95:.1f})")
    ax.axhline(thres_99, color="#e84855", linestyle="--", lw=1.5, label=f"99% ({thres_99:.1f})")
    ax.set_xticks(x_plot)
    ax.set_xticklabels(
        [str(label_pca[i])[-6:] for i in idx_sorted], rotation=90, fontsize=7
    )
    ax.set_ylabel("Hotelling T² Value")
    ax.set_title("Hotelling T² per Batch (sorted by time)")
    ax.legend(title="Confidence Level")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_contribution_bar(
    df_contrib: pd.DataFrame,
    title: str,
    x_col: str = "Contribution",
    y_col: str = "Feature",
) -> plt.Figure:
    """通用貢獻度水平條狀圖（正值紅、負值藍）。"""
    n = len(df_contrib)
    fig, ax = plt.subplots(figsize=(12, max(5, n * 0.4)))
    colors = ["#e84855" if v > 0 else "#2e86ab" for v in df_contrib[x_col]]
    ax.barh(df_contrib[y_col], df_contrib[x_col], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(x_col)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig
