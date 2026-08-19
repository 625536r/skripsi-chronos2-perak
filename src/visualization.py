"""Visualisasi hasil peramalan untuk laporan skripsi.

Seluruh gambar disimpan ke ``results/figures/`` dengan resolusi
``visualization.figure_dpi`` (300 dpi) dan berlabel bahasa Indonesia, siap
disisipkan ke BAB IV tanpa penyuntingan ulang.

Modul ini **tidak menghitung metrik apa pun sendiri**. Seluruh angka yang
digambar berasal dari :mod:`src.evaluation` atau dari berkas ramalan mentah
``results/forecasts/*.parquet`` — satu sumber kebenaran, sehingga gambar dan
tabel pada laporan tidak mungkin saling bertentangan.

Gambar yang dihasilkan:

1. :func:`plot_series_overview`      — ketiga seri ternormalisasi + batas split
2. :func:`plot_forecast_example`     — jendela uji terpilih, pita 80% dan 95%
3. :func:`plot_metrics_by_horizon`   — MAE dan CRPS terhadap h
4. :func:`plot_coverage_calibration` — reliability diagram (nominal vs empiris)
5. :func:`plot_error_distribution`   — boxplot galat absolut per skema

Cara menjalankan mandiri (menggambar ulang dari berkas yang sudah ada):
    python -m src.visualization
    python -m src.visualization --split val
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

# Backend non-interaktif: modul ini harus bisa berjalan di server tanpa display
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation import (  # noqa: E402
    aggregate_metrics,
    coverage,
    find_quantile_index,
    interval_indices,
)
from src.preprocessing import get_covariate_columns, get_target_column  # noqa: E402
from src.utils import load_config, resolve_path, setup_logger  # noqa: E402

# Palet dibuat sama persis dengan src/mutual_information.py supaya seluruh
# gambar pada laporan tampak berasal dari satu sistem visual.
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"

# Warna per skema. Baseline sengaja abu-abu: ia pembanding, bukan tokoh utama.
_SCHEME_COLORS: dict[str, str] = {
    "naive": "#898781",
    "zeroshot": "#2a78d6",
    "finetuned": "#eb6834",
}

# Label skema yang dipakai pada seluruh gambar dan tabel laporan
_SCHEME_LABELS: dict[str, str] = {
    "naive": "Naive Persistence",
    "zeroshot": "Chronos-2 Zero-Shot",
    "finetuned": "Chronos-2 Fine-Tuned",
}

# Urutan tampil skema: baseline dulu, lalu model
_SCHEME_ORDER: tuple[str, ...] = ("naive", "zeroshot", "finetuned")

# Warna deret harga pada gambar ikhtisar
_SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#3f8f5f")


class VisualizationError(ValueError):
    """Kegagalan pada pembentukan gambar."""


# =============================================================================
# Utilitas gaya
# =============================================================================


def _style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Menerapkan gaya sumbu yang seragam: grid tipis, spine minimal.

    Args:
        ax: Objek axes matplotlib yang akan digaya.
        grid_axis: Sumbu yang diberi garis bantu (``"x"``, ``"y"``, ``"both"``).
    """
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


def _save_figure(figure: plt.Figure, output_path: str | Path, dpi: int) -> Path:
    """Menyimpan gambar ke disk dan menutup figure agar memori tidak menumpuk.

    Args:
        figure: Objek figure matplotlib.
        output_path: Lokasi berkas keluaran.
        dpi: Resolusi gambar.

    Returns:
        Path absolut berkas gambar yang tersimpan.
    """
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def _scheme_label(scheme: str) -> str:
    """Mengambil label tampil sebuah skema.

    Args:
        scheme: Nama skema internal (``naive``/``zeroshot``/``finetuned``).

    Returns:
        Label berbahasa Indonesia untuk gambar dan tabel.
    """
    return _SCHEME_LABELS.get(scheme, scheme)


def _ordered_schemes(schemes: Any) -> list[str]:
    """Mengurutkan nama skema mengikuti urutan pelaporan yang baku.

    Args:
        schemes: Koleksi nama skema.

    Returns:
        Daftar nama skema terurut; skema tak dikenal ditaruh di belakang.
    """
    known = [scheme for scheme in _SCHEME_ORDER if scheme in schemes]
    unknown = sorted(scheme for scheme in schemes if scheme not in _SCHEME_ORDER)
    return known + unknown


def _short_label(column: str) -> str:
    """Membuat label pendek untuk gambar dari nama kolom.

    Args:
        column: Nama kolom, mis. ``"gold_close"``.

    Returns:
        Label ringkas, mis. ``"GOLD"``.
    """
    return column.replace("_close", "").upper()


# =============================================================================
# 1. Ikhtisar deret
# =============================================================================


def plot_series_overview(
    frames: dict[str, pd.DataFrame],
    output_path: str | Path,
    dpi: int,
    config: dict[str, Any] | None = None,
) -> Path:
    """Menggambar ketiga deret ternormalisasi beserta garis batas split.

    Ketiga harga berada pada satuan yang sangat berbeda (Perak puluhan dolar,
    Emas ribuan dolar, DXY sekitar 100), sehingga tidak dapat dibaca pada satu
    sumbu. Semuanya karena itu dinormalisasi ke basis
    ``visualization.normalization_base`` pada tanggal pertama:

        nilai_ternormalisasi_t = base * P_t / P_0

    Bentuk ini mempertahankan pergerakan relatif tiap deret sehingga arah
    hubungan Perak-Emas (searah) dan Perak-DXY (berlawanan) langsung terlihat.

    Dua garis vertikal menandai batas latih/validasi dan validasi/uji
    (kronologis, tanpa acak), sehingga pembaca dapat menilai sendiri apakah periode uji
    kebetulan jatuh pada rezim pasar yang tidak biasa.

    Args:
        frames: Dictionary ``{"train": df, "val": df, "test": df}`` dengan index
            tanggal.
        output_path: Lokasi berkas gambar keluaran.
        dpi: Resolusi gambar.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.

    Returns:
        Path absolut berkas gambar yang tersimpan.

    Raises:
        VisualizationError: Bila bagian data yang dibutuhkan tidak lengkap.
    """
    if config is None:
        config = load_config()

    missing = [name for name in ("train", "val", "test") if name not in frames]
    if missing:
        raise VisualizationError(f"Bagian data berikut tidak tersedia: {missing}.")

    base = config["visualization"]["normalization_base"]
    columns = [get_target_column(config), *get_covariate_columns(config)]

    combined = pd.concat([frames["train"], frames["val"], frames["test"]])
    combined = combined.sort_index()

    figure, ax = plt.subplots(figsize=(11.0, 5.0), facecolor=_SURFACE)
    _style_axes(ax)

    for color, column in zip(_SERIES_COLORS, columns):
        series = combined[column]
        normalized = base * series / series.iloc[0]
        ax.plot(
            normalized.index,
            normalized.to_numpy(),
            color=color,
            linewidth=1.3,
            label=_short_label(column),
            zorder=3,
        )

    # --- Garis batas split ---
    boundaries = (
        ("Batas latih / validasi", frames["val"].index[0]),
        ("Batas validasi / uji", frames["test"].index[0]),
    )
    y_top = ax.get_ylim()[1]
    for label, boundary_date in boundaries:
        ax.axvline(
            boundary_date, color=_INK_MUTED, linewidth=1.1, linestyle="--", zorder=2
        )
        ax.text(
            boundary_date,
            y_top,
            f" {label}",
            fontsize=8.5,
            color=_INK_MUTED,
            rotation=90,
            va="top",
            ha="left",
        )

    ax.set_xlabel("Tanggal perdagangan", fontsize=10, color=_INK_SECONDARY)
    ax.set_ylabel(
        f"Indeks harga (basis {base:g} pada {combined.index[0].date()})",
        fontsize=10,
        color=_INK_SECONDARY,
    )
    ax.set_title(
        "Pergerakan Perak, Emas, dan Dollar Index (ternormalisasi)",
        fontsize=12,
        color=_INK_PRIMARY,
        pad=12,
    )
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=_INK_SECONDARY)

    figure.text(
        0.5,
        -0.02,
        f"Latih {len(frames['train'])} hari  |  Validasi {len(frames['val'])} hari  |  "
        f"Uji {len(frames['test'])} hari  —  pembagian kronologis tanpa pengacakan",
        fontsize=8.5,
        color=_INK_MUTED,
        ha="center",
    )

    return _save_figure(figure, output_path, dpi)


# =============================================================================
# 2. Contoh jendela ramalan
# =============================================================================


def select_example_windows(n_windows: int, n_examples: int) -> list[int]:
    """Memilih indeks jendela contoh secara deterministik dan merata.

    Pemilihan sengaja TIDAK melihat galat: jendela diambil merata sepanjang
    periode uji (awal, tengah, akhir). Kalau jendela dipilih setelah melihat
    hasil, gambar pada laporan berubah menjadi etalase kasus terbaik dan
    kehilangan nilai buktinya.

    Args:
        n_windows: Banyaknya jendela yang tersedia.
        n_examples: Banyaknya jendela contoh yang diinginkan.

    Returns:
        Daftar indeks jendela terurut menaik.

    Raises:
        VisualizationError: Bila tidak ada jendela sama sekali.
    """
    if n_windows < 1:
        raise VisualizationError("Tidak ada jendela yang dapat digambar.")

    count = min(n_examples, n_windows)
    positions = np.linspace(0, n_windows - 1, count).round().astype(int)

    return sorted(set(int(position) for position in positions))


def plot_forecast_example(
    forecasts: dict[str, dict[str, Any]],
    output_path: str | Path,
    dpi: int,
    config: dict[str, Any] | None = None,
    schemes: tuple[str, ...] = ("zeroshot", "finetuned"),
    window_indices: list[int] | None = None,
) -> Path:
    """Menggambar beberapa jendela uji: aktual, median, serta pita 80% dan 95%.

    Tata letaknya baris = jendela, kolom = skema, sehingga zero-shot dan
    fine-tuned tampil **berdampingan pada jendela yang sama** dan pada sumbu-y
    yang identik per baris. Dengan begitu perbedaan lebar pita antar skema dapat
    dibandingkan langsung secara visual, bukan hanya lewat angka coverage.

    Args:
        forecasts: Dictionary ``{skema: keluaran rolling_forecast}``.
        output_path: Lokasi berkas gambar keluaran.
        dpi: Resolusi gambar.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        schemes: Skema yang digambar berdampingan.
        window_indices: Indeks jendela yang digambar. Bila ``None``, dipilih
            oleh :func:`select_example_windows`.

    Returns:
        Path absolut berkas gambar yang tersimpan.

    Raises:
        VisualizationError: Bila tidak ada satu pun skema yang tersedia.
    """
    if config is None:
        config = load_config()

    available = [scheme for scheme in schemes if scheme in forecasts]
    if not available:
        raise VisualizationError(
            f"Tidak ada skema {list(schemes)} pada hasil ramalan "
            f"(tersedia: {list(forecasts)})."
        )

    quantile_levels = config["model"]["quantile_levels"]
    median_index = find_quantile_index(
        quantile_levels, config["evaluation"]["point_quantile"]
    )
    low_80, high_80 = interval_indices(quantile_levels, 0.80)
    low_95, high_95 = interval_indices(quantile_levels, 0.95)

    reference = forecasts[available[0]]
    n_windows = reference["y_true"].shape[0]

    if window_indices is None:
        window_indices = select_example_windows(
            n_windows, config["visualization"]["n_example_windows"]
        )

    n_rows, n_columns = len(window_indices), len(available)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.4 * n_columns, 3.1 * n_rows),
        facecolor=_SURFACE,
        squeeze=False,
        sharex="row",
        sharey="row",
    )

    for row, window in enumerate(window_indices):
        for column, scheme in enumerate(available):
            ax = axes[row][column]
            _style_axes(ax)

            forecast = forecasts[scheme]
            actual = forecast["y_true"][window]
            quantiles = forecast["y_pred_quantiles"][window]
            steps = np.arange(1, len(actual) + 1)
            color = _SCHEME_COLORS.get(scheme, _INK_PRIMARY)

            ax.fill_between(
                steps,
                quantiles[:, low_95],
                quantiles[:, high_95],
                color=color,
                alpha=0.16,
                linewidth=0,
                label="Interval 95%",
                zorder=2,
            )
            ax.fill_between(
                steps,
                quantiles[:, low_80],
                quantiles[:, high_80],
                color=color,
                alpha=0.30,
                linewidth=0,
                label="Interval 80%",
                zorder=3,
            )
            ax.plot(
                steps,
                quantiles[:, median_index],
                color=color,
                linewidth=1.8,
                label="Median ramalan",
                zorder=4,
            )
            ax.plot(
                steps,
                actual,
                color=_INK_PRIMARY,
                linewidth=1.4,
                linestyle="--",
                marker="o",
                markersize=3.2,
                label="Aktual",
                zorder=5,
            )

            origin_date = pd.Timestamp(forecast["origin_dates"][window]).date()
            ax.set_title(
                f"{_scheme_label(scheme)} — origin {origin_date}",
                fontsize=10,
                color=_INK_PRIMARY,
                pad=8,
            )
            if row == n_rows - 1:
                ax.set_xlabel(
                    "Horizon h (hari perdagangan)", fontsize=9.5, color=_INK_SECONDARY
                )
            if column == 0:
                ax.set_ylabel("Harga Perak (USD)", fontsize=9.5, color=_INK_SECONDARY)
            ax.set_xticks(steps)

    axes[0][0].legend(
        frameon=False, fontsize=8.5, loc="best", labelcolor=_INK_SECONDARY, ncol=2
    )

    figure.suptitle(
        "Contoh jendela ramalan pada periode uji",
        fontsize=12.5,
        color=_INK_PRIMARY,
        y=1.0,
    )
    figure.tight_layout()

    return _save_figure(figure, output_path, dpi)


# =============================================================================
# 3. Metrik terhadap horizon
# =============================================================================


def plot_metrics_by_horizon(
    metrics: dict[str, dict[str, Any]],
    output_path: str | Path,
    dpi: int,
    config: dict[str, Any] | None = None,
) -> Path:
    """Menggambar MAE dan CRPS terhadap horizon h untuk seluruh skema.

    Gambar ini menjawab pertanyaan yang tidak dapat dijawab metrik agregat:
    pada horizon berapa keunggulan sebuah skema muncul atau hilang. Ramalan
    harga aset umumnya memburuk kira-kira sebanding akar horizon, sehingga
    kurva yang mendatar atau menurun justru patut dicurigai.

    Args:
        metrics: Dictionary ``{skema: keluaran aggregate_metrics}``.
        output_path: Lokasi berkas gambar keluaran.
        dpi: Resolusi gambar.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.

    Returns:
        Path absolut berkas gambar yang tersimpan.

    Raises:
        VisualizationError: Bila ``metrics`` kosong.
    """
    if config is None:
        config = load_config()
    if not metrics:
        raise VisualizationError("Tidak ada metrik yang dapat digambar.")

    panels = (("mae", "MAE (USD)"), ("crps", "CRPS (USD)"))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), facecolor=_SURFACE)

    for ax, (key, ylabel) in zip(axes, panels):
        _style_axes(ax)
        for scheme in _ordered_schemes(metrics):
            per_horizon = metrics[scheme]["per_horizon"]
            steps = [entry["h"] for entry in per_horizon]
            values = [entry[key] for entry in per_horizon]
            ax.plot(
                steps,
                values,
                color=_SCHEME_COLORS.get(scheme, _INK_PRIMARY),
                linewidth=1.8,
                marker="o",
                markersize=4.0,
                markeredgecolor=_SURFACE,
                markeredgewidth=0.8,
                label=_scheme_label(scheme),
                zorder=3,
            )
        ax.set_xlabel("Horizon h (hari perdagangan)", fontsize=10, color=_INK_SECONDARY)
        ax.set_ylabel(ylabel, fontsize=10, color=_INK_SECONDARY)
        ax.set_xticks(steps)

    axes[0].set_title(
        "Akurasi titik", fontsize=11, color=_INK_PRIMARY, pad=10
    )
    axes[1].set_title(
        "Kualitas sebaran prediktif", fontsize=11, color=_INK_PRIMARY, pad=10
    )
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=_INK_SECONDARY)

    figure.suptitle(
        "Pemburukan akurasi seiring bertambahnya horizon",
        fontsize=12.5,
        color=_INK_PRIMARY,
        y=1.02,
    )
    figure.tight_layout()

    return _save_figure(figure, output_path, dpi)


# =============================================================================
# 4. Kalibrasi cakupan (reliability diagram)
# =============================================================================


def compute_calibration(
    forecast: dict[str, Any],
    calibration_levels: list[float],
    quantile_levels: list[float],
) -> list[dict[str, float]]:
    """Menghitung cakupan empiris untuk sederet tingkat cakupan nominal.

    Tiap tingkat nominal ``c`` dibentuk dari pasangan kuantil ``(1-c)/2`` dan
    ``1-(1-c)/2`` (lihat :func:`src.evaluation.interval_indices`), lalu cakupan
    empirisnya dihitung ulang memakai :func:`src.evaluation.coverage`.

    Args:
        forecast: Keluaran rolling forecast satu skema.
        calibration_levels: Daftar tingkat cakupan nominal yang diuji.
        quantile_levels: Daftar level kuantil ramalan.

    Returns:
        Daftar dictionary ``{"nominal": ..., "empirical": ..., "width": ...}``.
    """
    actual = forecast["y_true"]
    quantiles = forecast["y_pred_quantiles"]

    points: list[dict[str, float]] = []
    for level in calibration_levels:
        lower_index, upper_index = interval_indices(quantile_levels, level)
        lower = quantiles[:, :, lower_index]
        upper = quantiles[:, :, upper_index]
        points.append(
            {
                "nominal": float(level),
                "empirical": coverage(actual, lower, upper),
                "width": float((upper - lower).mean()),
            }
        )

    return points


def plot_coverage_calibration(
    forecasts: dict[str, dict[str, Any]],
    output_path: str | Path,
    dpi: int,
    config: dict[str, Any] | None = None,
) -> Path:
    """Menggambar reliability diagram: cakupan nominal vs cakupan empiris.

    Panel kiri adalah diagram kalibrasi itu sendiri. Titik di **bawah** garis
    diagonal berarti interval terlalu sempit (model terlalu percaya diri), di
    **atas** berarti terlalu lebar (terlalu berhati-hati).

    Panel kanan menampilkan lebar interval rata-rata pada tiap tingkat nominal,
    dan wajib dibaca bersama panel kiri: cakupan yang tepat sasaran tidak ada
    artinya bila dicapai dengan interval yang jauh lebih lebar daripada skema
    pembanding.

    Args:
        forecasts: Dictionary ``{skema: keluaran rolling_forecast}``.
        output_path: Lokasi berkas gambar keluaran.
        dpi: Resolusi gambar.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.

    Returns:
        Path absolut berkas gambar yang tersimpan.

    Raises:
        VisualizationError: Bila ``forecasts`` kosong.
    """
    if config is None:
        config = load_config()
    if not forecasts:
        raise VisualizationError("Tidak ada ramalan yang dapat digambar.")

    quantile_levels = config["model"]["quantile_levels"]
    calibration_levels = config["visualization"]["calibration_levels"]

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), facecolor=_SURFACE)
    for ax in axes:
        _style_axes(ax, grid_axis="both")

    # --- Panel kiri: diagram kalibrasi ---
    axes[0].plot(
        [0, 1],
        [0, 1],
        color=_INK_MUTED,
        linewidth=1.1,
        linestyle="--",
        label="Kalibrasi sempurna",
        zorder=2,
    )

    for scheme in _ordered_schemes(forecasts):
        points = compute_calibration(
            forecasts[scheme], calibration_levels, quantile_levels
        )
        nominal = [point["nominal"] for point in points]
        empirical = [point["empirical"] for point in points]
        widths = [point["width"] for point in points]
        color = _SCHEME_COLORS.get(scheme, _INK_PRIMARY)

        axes[0].plot(
            nominal,
            empirical,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4.5,
            markeredgecolor=_SURFACE,
            markeredgewidth=0.8,
            label=_scheme_label(scheme),
            zorder=3,
        )
        axes[1].plot(
            nominal,
            widths,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4.5,
            markeredgecolor=_SURFACE,
            markeredgewidth=0.8,
            label=_scheme_label(scheme),
            zorder=3,
        )

    axes[0].set_xlabel("Cakupan nominal", fontsize=10, color=_INK_SECONDARY)
    axes[0].set_ylabel("Cakupan empiris", fontsize=10, color=_INK_SECONDARY)
    axes[0].set_title(
        "Reliability diagram", fontsize=11, color=_INK_PRIMARY, pad=10
    )
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False, fontsize=9.0, loc="upper left", labelcolor=_INK_SECONDARY)

    axes[1].set_xlabel("Cakupan nominal", fontsize=10, color=_INK_SECONDARY)
    axes[1].set_ylabel("Lebar interval rata-rata (USD)", fontsize=10, color=_INK_SECONDARY)
    axes[1].set_title(
        "Lebar interval pada tiap tingkat", fontsize=11, color=_INK_PRIMARY, pad=10
    )

    figure.text(
        0.5,
        -0.04,
        "Titik di bawah diagonal = interval terlalu sempit (model terlalu percaya diri); "
        "di atas diagonal = terlalu lebar.",
        fontsize=8.5,
        color=_INK_MUTED,
        ha="center",
    )
    figure.suptitle(
        "Kalibrasi interval prediksi pada periode uji",
        fontsize=12.5,
        color=_INK_PRIMARY,
        y=1.02,
    )
    figure.tight_layout()

    return _save_figure(figure, output_path, dpi)


# =============================================================================
# 5. Sebaran galat
# =============================================================================


def plot_error_distribution(
    forecasts: dict[str, dict[str, Any]],
    output_path: str | Path,
    dpi: int,
    config: dict[str, Any] | None = None,
) -> Path:
    """Menggambar boxplot galat absolut per skema.

    Metrik agregat seperti MAE hanya melaporkan pusat sebaran. Boxplot ini
    memperlihatkan seluruh sebarannya: median, rentang antar-kuartil, dan ekor
    galat besar. Dua skema dengan MAE hampir sama bisa saja memiliki perilaku
    ekor yang sangat berbeda — dan justru ekor itulah yang penting pada
    peramalan harga.

    Panel kanan memecah sebaran menurut horizon untuk skema terbaik menurut
    MAE, sehingga terlihat bagaimana ekor galat melebar seiring bertambahnya h.

    Args:
        forecasts: Dictionary ``{skema: keluaran rolling_forecast}``.
        output_path: Lokasi berkas gambar keluaran.
        dpi: Resolusi gambar.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.

    Returns:
        Path absolut berkas gambar yang tersimpan.

    Raises:
        VisualizationError: Bila ``forecasts`` kosong.
    """
    if config is None:
        config = load_config()
    if not forecasts:
        raise VisualizationError("Tidak ada ramalan yang dapat digambar.")

    schemes = _ordered_schemes(forecasts)
    errors = {
        scheme: np.abs(
            forecasts[scheme]["y_true"] - forecasts[scheme]["y_pred"]
        )
        for scheme in schemes
    }

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), facecolor=_SURFACE)
    for ax in axes:
        _style_axes(ax)

    # --- Panel kiri: sebaran seluruh galat per skema ---
    boxes = axes[0].boxplot(
        [errors[scheme].ravel() for scheme in schemes],
        tick_labels=[_scheme_label(scheme) for scheme in schemes],
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 2.4,
            "markerfacecolor": _INK_MUTED,
            "markeredgecolor": "none",
            "alpha": 0.35,
        },
        medianprops={"color": _INK_PRIMARY, "linewidth": 1.4},
        whiskerprops={"color": _BASELINE, "linewidth": 1.0},
        capprops={"color": _BASELINE, "linewidth": 1.0},
    )
    for patch, scheme in zip(boxes["boxes"], schemes):
        patch.set_facecolor(_SCHEME_COLORS.get(scheme, _INK_MUTED))
        patch.set_alpha(0.45)
        patch.set_edgecolor(_SCHEME_COLORS.get(scheme, _INK_MUTED))
        patch.set_linewidth(1.1)

    # Rata-rata (= MAE) ditandai agar boxplot dapat dicocokkan dengan tabel
    for position, scheme in enumerate(schemes, start=1):
        axes[0].scatter(
            position,
            errors[scheme].mean(),
            marker="D",
            s=22,
            color=_INK_PRIMARY,
            zorder=5,
            label="Rata-rata (MAE)" if position == 1 else None,
        )

    axes[0].set_ylabel("Galat absolut (USD)", fontsize=10, color=_INK_SECONDARY)
    axes[0].set_title(
        "Sebaran galat seluruh jendela dan horizon",
        fontsize=11,
        color=_INK_PRIMARY,
        pad=10,
    )
    axes[0].legend(frameon=False, fontsize=9.0, loc="upper left", labelcolor=_INK_SECONDARY)
    axes[0].tick_params(axis="x", labelrotation=12)

    # --- Panel kanan: pemecahan menurut horizon untuk skema terbaik ---
    best_scheme = min(schemes, key=lambda scheme: errors[scheme].mean())
    best_errors = errors[best_scheme]
    horizon = best_errors.shape[1]

    horizon_boxes = axes[1].boxplot(
        [best_errors[:, step] for step in range(horizon)],
        tick_labels=[str(step + 1) for step in range(horizon)],
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": _INK_PRIMARY, "linewidth": 1.3},
        whiskerprops={"color": _BASELINE, "linewidth": 1.0},
        capprops={"color": _BASELINE, "linewidth": 1.0},
    )
    color = _SCHEME_COLORS.get(best_scheme, _INK_MUTED)
    for patch in horizon_boxes["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.40)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.0)

    axes[1].set_xlabel("Horizon h (hari perdagangan)", fontsize=10, color=_INK_SECONDARY)
    axes[1].set_ylabel("Galat absolut (USD)", fontsize=10, color=_INK_SECONDARY)
    axes[1].set_title(
        f"Sebaran galat per horizon — {_scheme_label(best_scheme)} (MAE terkecil)",
        fontsize=11,
        color=_INK_PRIMARY,
        pad=10,
    )

    figure.suptitle(
        "Sebaran galat absolut pada periode uji",
        fontsize=12.5,
        color=_INK_PRIMARY,
        y=1.02,
    )
    figure.tight_layout()

    return _save_figure(figure, output_path, dpi)


# =============================================================================
# Orkestrasi seluruh gambar
# =============================================================================


def generate_all_figures(
    forecasts: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]] | None = None,
    frames: dict[str, pd.DataFrame] | None = None,
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    """Menghasilkan seluruh gambar laporan sekaligus.

    Gambar yang bahan bakunya tidak tersedia dilewati dengan peringatan, bukan
    menggagalkan seluruh pipeline — misalnya ketika hanya skema zero-shot yang
    dijalankan sehingga contoh jendela fine-tuned belum ada.

    Args:
        forecasts: Dictionary ``{skema: keluaran rolling_forecast}``.
        metrics: Dictionary ``{skema: keluaran aggregate_metrics}``. Bila
            ``None``, dihitung ulang dari ``forecasts``.
        frames: Dictionary bagian data untuk gambar ikhtisar. Bila ``None``,
            gambar ikhtisar dilewati.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dictionary ``{nama_gambar: path}`` untuk gambar yang berhasil dibuat.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger(
            "visualization", log_dir=config.get("paths", {}).get("logs_dir", "logs")
        )

    dpi = config["visualization"]["figure_dpi"]
    figures_dir = resolve_path(config["paths"]["figures_dir"])
    quantile_levels = config["model"]["quantile_levels"]

    if metrics is None:
        naive_point = forecasts.get("naive", {}).get("y_pred")
        metrics = {
            scheme: aggregate_metrics(
                forecast["y_true"],
                forecast["y_pred_quantiles"],
                quantile_levels,
                y_naive=naive_point,
                config=config,
            )
            for scheme, forecast in forecasts.items()
        }

    tasks = [
        (
            "series_overview",
            "01_ikhtisar_deret.png",
            lambda path: plot_series_overview(frames, path, dpi, config=config),
            frames is not None,
        ),
        (
            "forecast_example",
            "02_contoh_ramalan.png",
            lambda path: plot_forecast_example(forecasts, path, dpi, config=config),
            bool(forecasts),
        ),
        (
            "metrics_by_horizon",
            "03_metrik_per_horizon.png",
            lambda path: plot_metrics_by_horizon(metrics, path, dpi, config=config),
            bool(metrics),
        ),
        (
            "coverage_calibration",
            "04_kalibrasi_cakupan.png",
            lambda path: plot_coverage_calibration(forecasts, path, dpi, config=config),
            bool(forecasts),
        ),
        (
            "error_distribution",
            "05_sebaran_galat.png",
            lambda path: plot_error_distribution(forecasts, path, dpi, config=config),
            bool(forecasts),
        ),
    ]

    saved: dict[str, Path] = {}
    for name, filename, builder, available in tasks:
        if not available:
            logger.warning("Gambar '%s' dilewati: bahan bakunya tidak tersedia.", name)
            continue
        try:
            path = builder(figures_dir / filename)
            saved[name] = path
            logger.info("Gambar '%s' tersimpan -> %s", name, path)
        except (VisualizationError, KeyError, ValueError) as error:
            logger.warning(
                "Gambar '%s' gagal dibuat: %s: %s", name, type(error).__name__, error
            )

    return saved


# =============================================================================
# Titik masuk mandiri
# =============================================================================


def main(argv: list[str] | None = None) -> dict[str, Path]:
    """Titik masuk eksekusi mandiri: menggambar ulang dari berkas yang ada.

    Berguna untuk menyesuaikan tampilan gambar tanpa menjalankan model lagi —
    seluruh bahan diambil dari ``results/forecasts/*.parquet``.

    Args:
        argv: Daftar argumen baris perintah.

    Returns:
        Dictionary ``{nama_gambar: path}``.
    """
    parser = argparse.ArgumentParser(
        description="Menggambar ulang seluruh gambar laporan dari berkas ramalan."
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Bagian data sumber ramalan (bawaan: test).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Panjang horizon H. Bawaan: model.prediction_length pada config.",
    )
    args = parser.parse_args(argv)

    from src.forecasting import (  # impor lokal: hindari biaya impor AutoGluon
        ALL_SCHEMES,
        forecast_file_path,
        load_forecast_arrays,
        load_processed_frames,
    )

    config = load_config()
    logger = setup_logger(
        "visualization", log_dir=config.get("paths", {}).get("logs_dir", "logs")
    )
    horizon = args.horizon or config["model"]["prediction_length"]

    forecasts: dict[str, dict[str, Any]] = {}
    for scheme in ALL_SCHEMES:
        if forecast_file_path(scheme, horizon, args.split, config).is_file():
            forecasts[scheme] = load_forecast_arrays(
                scheme, horizon, args.split, config
            )
            logger.info("Memuat ramalan skema '%s'.", scheme)

    if not forecasts:
        raise VisualizationError(
            f"Tidak ada berkas ramalan untuk H={horizon}, split='{args.split}'. "
            f"Jalankan 'python run_pipeline.py' terlebih dahulu."
        )

    frames = load_processed_frames(("train", "val", "test"), config=config, logger=logger)

    return generate_all_figures(
        forecasts, frames=frames, config=config, logger=logger
    )


if __name__ == "__main__":
    main()
