"""Dua gambar tambahan untuk bimbingan — tidak menjalankan model apa pun.

Sumber data:
    - results/metrics/comparison_table.md / final_results.json (Tabel 1 & uji DM)
    - results/forecasts/*_H10.parquet (uji ketahanan tanpa 2026-01-30)

Gaya visual disamakan persis dengan src/visualization.py agar tampak berasal
dari satu sistem. Keluaran:
    - results/figures/bimbingan/ringkasan_tabel1.png
    - results/figures/bimbingan/ketahanan_2026-01-30.png

Jalankan: python catatan/gambar_ringkasan_bimbingan.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- Palet & gaya, disalin dari src/visualization.py agar konsisten ---------
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"

_SCHEME_COLORS = {"naive": "#898781", "zeroshot": "#2a78d6", "finetuned": "#eb6834"}
_SCHEME_LABELS = {
    "naive": "Naive Persistence",
    "zeroshot": "Chronos-2 Zero-Shot",
    "finetuned": "Chronos-2 Fine-Tuned",
}
_SCHEME_ORDER = ("naive", "zeroshot", "finetuned")
_DPI = 300


def _style_axes(ax, grid_axis="y"):
    ax.set_facecolor(_SURFACE)
    ax.grid(axis=grid_axis, color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=_INK_MUTED, labelsize=9, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(_INK_SECONDARY)


def _save(fig, path):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("tersimpan:", path)


# =============================================================================
# Gambar 1 — Ringkasan Tabel 1 (MAE/CRPS/Cov80/Width80 + p-value uji DM)
# =============================================================================


def build_overview_figure():
    final = json.loads((ROOT / "results/metrics/final_results.json").read_text(encoding="utf-8"))
    metrics = final["metrics"]
    dm = {f"{row['scheme_a']}_vs_{row['scheme_b']}": row for row in final["diebold_mariano"]}

    schemes = [s for s in _SCHEME_ORDER if s in metrics]
    mae = [metrics[s]["overall"]["mae"] for s in schemes]
    crps = [metrics[s]["overall"]["crps"] for s in schemes]
    cov80 = [metrics[s]["overall"]["coverage"]["0.8"] for s in schemes]
    width80 = [metrics[s]["overall"]["interval_width"]["0.8"] for s in schemes]

    x = np.arange(len(schemes))
    colors = [_SCHEME_COLORS[s] for s in schemes]
    labels = [_SCHEME_LABELS[s] for s in schemes]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor=_SURFACE)

    # --- Panel 1: MAE & CRPS berdampingan ---
    ax = axes[0]
    _style_axes(ax)
    w = 0.35
    bars_mae = ax.bar(x - w / 2, mae, width=w, color=colors, alpha=1.0, zorder=3, label="MAE")
    bars_crps = ax.bar(
        x + w / 2, crps, width=w, color=colors, alpha=0.45, zorder=3, label="CRPS"
    )
    for b in list(bars_mae) + list(bars_crps):
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.06, f"{b.get_height():.2f}",
            ha="center", va="bottom", fontsize=8.2, color=_INK_SECONDARY,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.6, rotation=12)
    ax.set_ylabel("USD", fontsize=10, color=_INK_SECONDARY)
    ax.set_title("Akurasi: MAE (solid) & CRPS (pudar)", fontsize=10.8, color=_INK_PRIMARY, pad=10)

    # --- Panel 2: Cov80 vs target ---
    ax = axes[1]
    _style_axes(ax)
    bars = ax.bar(x, cov80, width=0.5, color=colors, zorder=3)
    ax.axhline(0.80, color=_INK_MUTED, linestyle="--", linewidth=1.1, zorder=2)
    ax.text(
        0.98, 0.815, "target 0,80", fontsize=8.2, color=_INK_MUTED, va="bottom", ha="right",
        transform=ax.get_yaxis_transform(),
    )
    for b, v, wd in zip(bars, cov80, width80):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
            ha="center", va="bottom", fontsize=8.6, color=_INK_SECONDARY,
        )
        ax.text(
            b.get_x() + b.get_width() / 2, 0.02, f"lebar {wd:.1f}",
            ha="center", va="bottom", fontsize=7.6, color=_SURFACE, zorder=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.6, rotation=12)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Cakupan empiris 80%", fontsize=10, color=_INK_SECONDARY)
    ax.set_title("Kalibrasi interval 80%", fontsize=10.8, color=_INK_PRIMARY, pad=10)

    # --- Panel 3: uji DM utama, p-value ---
    ax = axes[2]
    _style_axes(ax, grid_axis="x")
    dm_pairs = [
        ("zeroshot_vs_finetuned", "Zero-Shot vs\nFine-Tuned"),
        ("zeroshot_vs_naive", "Zero-Shot vs\nNaive"),
        ("finetuned_vs_naive", "Fine-Tuned vs\nNaive"),
    ]
    p_values = [dm[key]["p_value"] for key, _ in dm_pairs]
    y_pos = np.arange(len(dm_pairs))
    bar_colors = ["#c3512e" if p < 0.05 else _INK_MUTED for p in p_values]
    ax.barh(y_pos, p_values, color=bar_colors, zorder=3, height=0.55)
    ax.axvline(0.05, color=_INK_PRIMARY, linestyle="--", linewidth=1.1, zorder=2)
    ax.text(0.052, -0.65, "α = 0,05", fontsize=8.2, color=_INK_PRIMARY, va="bottom")
    for y, p in zip(y_pos, p_values):
        ax.text(p + 0.02, y, f"p = {p:.4f}", va="center", fontsize=8.6, color=_INK_SECONDARY)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl for _, lbl in dm_pairs], fontsize=8.6)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(len(dm_pairs) - 0.5, -1.0)
    ax.set_xlabel("p-value uji Diebold-Mariano", fontsize=10, color=_INK_SECONDARY)
    ax.set_title("Tidak ada perbedaan yang terbukti", fontsize=10.8, color=_INK_PRIMARY, pad=10)

    fig.suptitle(
        "Ringkasan Tabel 1 — perbandingan skema pada periode uji (181 jendela, H=10)",
        fontsize=13,
        color=_INK_PRIMARY,
        y=1.04,
    )
    fig.text(
        0.5, -0.04,
        "Batang pudar pada panel kiri = CRPS. Batang merah pada panel kanan = signifikan (tidak ada).",
        fontsize=8.3, color=_INK_MUTED, ha="center",
    )
    fig.tight_layout()
    _save(fig, "results/figures/bimbingan/ringkasan_tabel1.png")


# =============================================================================
# Gambar 2 — Uji ketahanan tanpa jendela yang memuat 2026-01-30
# =============================================================================

EVENT = "2026-01-30"
H = 10
QL = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]
QCOLS = [f"q{q}" for q in QL]


def _load(scheme):
    df = pd.read_parquet(ROOT / f"results/forecasts/{scheme}_H10.parquet")
    df["target_date"] = df["target_date"].astype(str)
    return df.sort_values(["window", "h"]).reset_index(drop=True)


def _grid(df, windows, col):
    sub = df[df["window"].isin(windows)]
    return sub.pivot(index="window", columns="h", values=col).to_numpy()


def build_robustness_figure():
    data = {s: _load(s) for s in _SCHEME_ORDER}
    zs = data["zeroshot"]
    hit = sorted(zs.loc[zs["target_date"] == EVENT, "window"].unique())
    all_w = sorted(zs["window"].unique())
    clean_w = [w for w in all_w if w not in hit]

    def metrics_for(scheme, windows):
        df = data[scheme]
        y = _grid(df, windows, "y_true")
        med = _grid(df, windows, "q0.5")
        q = np.stack([_grid(df, windows, c) for c in QCOLS], axis=-1)
        lo, hi = q[..., QL.index(0.1)], q[..., QL.index(0.9)]
        return {
            "mae": float(np.abs(y - med).mean()),
            "cov80": float(((y >= lo) & (y <= hi)).mean()),
        }

    subsets = {"Semua jendela\n(n=181)": all_w, "Tanpa peristiwa\n(n=171)": clean_w}
    schemes = list(_SCHEME_ORDER)
    colors = [_SCHEME_COLORS[s] for s in schemes]
    labels = [_SCHEME_LABELS[s] for s in schemes]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), facecolor=_SURFACE)

    x = np.arange(len(subsets))
    w = 0.24

    # --- Panel kiri: MAE ---
    ax = axes[0]
    _style_axes(ax)
    for i, s in enumerate(schemes):
        vals = [metrics_for(s, wins)["mae"] for wins in subsets.values()]
        bars = ax.bar(x + (i - 1) * w, vals, width=w, color=colors[i], zorder=3, label=labels[i])
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.8, color=_INK_SECONDARY,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(list(subsets.keys()), fontsize=9.2)
    ax.set_ylabel("MAE (USD)", fontsize=10, color=_INK_SECONDARY)
    ax.set_title("MAE: dengan vs tanpa 2026-01-30", fontsize=10.8, color=_INK_PRIMARY, pad=10)
    handles, legend_labels = ax.get_legend_handles_labels()

    # --- Panel kanan: Cov80 ---
    ax = axes[1]
    _style_axes(ax)
    for i, s in enumerate(schemes):
        vals = [metrics_for(s, wins)["cov80"] for wins in subsets.values()]
        bars = ax.bar(x + (i - 1) * w, vals, width=w, color=colors[i], zorder=3, label=labels[i])
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.8, color=_INK_SECONDARY,
            )
    ax.axhline(0.80, color=_INK_MUTED, linestyle="--", linewidth=1.1, zorder=2)
    ax.text(x[-1] + 0.3, 0.805, "target 0,80", fontsize=8.0, color=_INK_MUTED, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(list(subsets.keys()), fontsize=9.2)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Cakupan empiris 80%", fontsize=10, color=_INK_SECONDARY)
    ax.set_title("Cov80: dengan vs tanpa 2026-01-30", fontsize=10.8, color=_INK_PRIMARY, pad=10)

    fig.legend(
        handles, legend_labels, frameon=False, fontsize=8.6, loc="upper center",
        ncol=3, bbox_to_anchor=(0.5, 1.10), labelcolor=_INK_SECONDARY,
    )
    fig.suptitle(
        "Uji ketahanan — 10 jendela yang memuat penurunan 37,61% pada 2026-01-30",
        fontsize=12.6,
        color=_INK_PRIMARY,
        y=1.18,
    )
    fig.text(
        0.5, -0.04,
        "Kesimpulan uji DM tidak berubah setelah peristiwa dikeluarkan; kalibrasi naive justru "
        "tepat sasaran (Cov80 = 0,801) begitu peristiwa itu disaring.",
        fontsize=8.2, color=_INK_MUTED, ha="center", wrap=True,
    )
    fig.tight_layout()
    _save(fig, "results/figures/bimbingan/ketahanan_2026-01-30.png")


if __name__ == "__main__":
    build_overview_figure()
    build_robustness_figure()
