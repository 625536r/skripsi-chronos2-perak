"""Analisis Mutual Information (MI) antara harga Perak dan kovariatnya.

Modul ini mengimplementasikan keputusan metodologis D6 pada CLAUDE.md:

* MI dihitung **hanya** dari data latih (``data/processed/train.csv``). Data
  validasi maupun data uji tidak pernah disentuh di sini, sehingga pemilihan
  kovariat tidak menimbulkan kebocoran informasi (aturan kerja no. 2).
* Dua varian wajib dilaporkan:
    - ``level``       : harga apa adanya.
    - ``log_return``  : ``log(P_t / P_{t-1})``, baris pertama dibuang.
  Varian log-return adalah varian yang secara statistik sah, karena MI pada
  level harga didominasi tren bersama sehingga cenderung overestimasi.
* Sebagai pembanding konteks, korelasi Pearson dan Spearman dihitung untuk
  kedua varian.
* Signifikansi MI diuji dengan permutation test.

Estimator yang dipakai adalah ``sklearn.feature_selection.mutual_info_regression``
(estimator k-NN Kraskov). Estimator ini stokastik (menambahkan derau kecil untuk
memecah nilai kembar), sehingga setiap angka MI dihitung ulang beberapa kali
dengan ``random_state`` berbeda lalu dilaporkan sebagai rata-rata +/- simpangan
baku.

Cara menjalankan mandiri:
    python -m src.mutual_information

Keluaran:
    results/metrics/mutual_information.json
    results/figures/mi_barplot.png
    results/figures/mi_lag_plot.png
    Tabel ringkas format markdown dicetak ke stdout.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

# Backend non-interaktif: modul harus bisa jalan tanpa display (mis. di server)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from src.data_collection import DATE_INDEX_NAME
from src.preprocessing import get_covariate_columns, get_target_column
from src.utils import (
    PROJECT_ROOT,
    load_config,
    resolve_path,
    save_json,
    setup_logger,
)

# Nama varian kanonis (harus cocok dengan config: mutual_information.variants)
MODE_LEVEL = "level"
MODE_LOG_RETURN = "log_return"

# Alias yang diterima agar pemanggilan "logreturn"/"log-return" tidak error
_MODE_ALIASES: dict[str, str] = {
    "level": MODE_LEVEL,
    "price": MODE_LEVEL,
    "log_return": MODE_LOG_RETURN,
    "logreturn": MODE_LOG_RETURN,
    "log-return": MODE_LOG_RETURN,
    "return": MODE_LOG_RETURN,
}

# Label yang dipakai pada tabel dan gambar
MODE_LABELS: dict[str, str] = {
    MODE_LEVEL: "Level harga",
    MODE_LOG_RETURN: "Log-return",
}

# --- Palet gambar (light surface, lolos pemeriksaan keterbacaan CVD) ---------
_SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834")  # biru, oranye
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"


class MutualInformationError(ValueError):
    """Kegagalan pada tahap perhitungan mutual information."""


# =============================================================================
# Fungsi inti
# =============================================================================


def _standardize(values: np.ndarray) -> np.ndarray:
    """Menstandardisasi sebuah vektor menjadi rata-rata 0 dan simpangan baku 1.

    Standardisasi tidak mengubah nilai MI secara teoretis (MI invarian terhadap
    transformasi monoton), tetapi menyeragamkan skala derau internal yang
    ditambahkan estimator k-NN sehingga hasilnya lebih stabil dan sebanding
    antar-pasangan variabel.

    Args:
        values: Vektor satu dimensi.

    Returns:
        Vektor terstandardisasi.

    Raises:
        MutualInformationError: Bila vektor konstan (simpangan baku nol).
    """
    array = np.asarray(values, dtype=float).ravel()
    std = array.std(ddof=0)
    if not np.isfinite(std) or std == 0.0:
        raise MutualInformationError(
            "Vektor konstan atau mengandung nilai tidak berhingga, "
            "MI tidak dapat dihitung."
        )
    return (array - array.mean()) / std


def _as_clean_pair(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Menyelaraskan dua vektor dan membuang pasangan yang mengandung NaN.

    Args:
        x: Vektor pertama (array-like).
        y: Vektor kedua (array-like).

    Returns:
        Pasangan array yang panjangnya sama dan bebas NaN.

    Raises:
        MutualInformationError: Bila panjang berbeda atau sisa data kosong.
    """
    x_array = np.asarray(x, dtype=float).ravel()
    y_array = np.asarray(y, dtype=float).ravel()

    if x_array.shape != y_array.shape:
        raise MutualInformationError(
            f"Panjang x ({x_array.size}) dan y ({y_array.size}) tidak sama."
        )

    mask = np.isfinite(x_array) & np.isfinite(y_array)
    if not mask.any():
        raise MutualInformationError("Tidak ada pasangan data valid setelah membuang NaN.")

    return x_array[mask], y_array[mask]


def compute_mi(x: Any, y: Any, n_neighbors: int, random_state: int) -> float:
    """Menghitung mutual information antara dua vektor kontinu.

    Kedua vektor distandardisasi terlebih dahulu, lalu MI diestimasi dengan
    ``mutual_info_regression`` (estimator k-NN). Nilai negatif kecil yang bisa
    muncul akibat bias estimator dipangkas menjadi nol, karena MI secara
    teoretis tidak pernah negatif.

    Args:
        x: Vektor prediktor (array-like satu dimensi).
        y: Vektor target (array-like satu dimensi).
        n_neighbors: Jumlah tetangga terdekat pada estimator k-NN.
        random_state: Seed estimator (mengendalikan derau pemecah nilai kembar).

    Returns:
        Estimasi MI dalam satuan nat (non-negatif).
    """
    x_clean, y_clean = _as_clean_pair(x, y)

    mi = mutual_info_regression(
        _standardize(x_clean).reshape(-1, 1),
        _standardize(y_clean),
        discrete_features=False,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )[0]

    return float(max(mi, 0.0))


def compute_mi_stable(
    x: Any,
    y: Any,
    n_repeats: int = 20,
    *,
    n_neighbors: int,
    random_state: int,
) -> tuple[float, float]:
    """Menghitung MI berulang kali untuk meredam sifat stokastik estimator k-NN.

    Estimator k-NN menambahkan derau acak kecil pada data sehingga satu kali
    pemanggilan bisa memberi hasil sedikit berbeda. Fungsi ini mengulang
    perhitungan dengan ``random_state`` berurutan (``random_state``,
    ``random_state + 1``, ...) lalu melaporkan rata-rata dan simpangan bakunya.

    Args:
        x: Vektor prediktor.
        y: Vektor target.
        n_repeats: Banyaknya pengulangan.
        n_neighbors: Jumlah tetangga terdekat pada estimator k-NN.
        random_state: Seed awal; pengulangan ke-i memakai ``random_state + i``.

    Returns:
        Tuple ``(mean, std)`` dari nilai MI hasil pengulangan.

    Raises:
        MutualInformationError: Bila ``n_repeats`` kurang dari satu.
    """
    if n_repeats < 1:
        raise MutualInformationError(f"n_repeats harus >= 1, diterima {n_repeats}.")

    x_clean, y_clean = _as_clean_pair(x, y)

    values = np.array(
        [
            compute_mi(
                x_clean, y_clean, n_neighbors=n_neighbors, random_state=random_state + i
            )
            for i in range(n_repeats)
        ],
        dtype=float,
    )

    return float(values.mean()), float(values.std(ddof=1) if n_repeats > 1 else 0.0)


def permutation_test(
    x: Any,
    y: Any,
    n_permutations: int,
    random_state: int,
    *,
    n_neighbors: int,
) -> float:
    """Menguji signifikansi MI dengan permutasi (uji acak).

    Hipotesis nol: ``x`` dan ``y`` saling bebas. Distribusi nol dibentuk dengan
    mengacak urutan ``y`` sebanyak ``n_permutations`` kali dan menghitung MI tiap
    kali. Nilai p adalah proporsi ``MI_null >= MI_observed``, dengan koreksi
    tambah-satu (``(1 + jumlah) / (1 + n_permutations)``) sehingga nilai p tidak
    pernah tepat nol dan tetap valid sebagai estimator.

    Catatan penting untuk varian ``level``: permutasi mengasumsikan pengamatan
    saling bebas, padahal harga sangat berautokorelasi. Nilai p pada varian level
    karena itu bersifat anti-konservatif (terlalu optimistis) dan hanya dipakai
    sebagai indikasi, bukan bukti formal. Pada varian log-return asumsi ini jauh
    lebih masuk akal.

    Args:
        x: Vektor prediktor.
        y: Vektor target yang akan diacak.
        n_permutations: Banyaknya permutasi.
        random_state: Seed generator permutasi.
        n_neighbors: Jumlah tetangga terdekat pada estimator k-NN.

    Returns:
        Nilai p hasil uji permutasi.
    """
    return float(
        _permutation_test_detail(
            x,
            y,
            n_permutations=n_permutations,
            random_state=random_state,
            n_neighbors=n_neighbors,
        )["p_value"]
    )


def permutation_null_distribution(
    x: Any,
    y: Any,
    n_permutations: int,
    random_state: int,
    *,
    n_neighbors: int,
) -> tuple[float, np.ndarray]:
    """Membentuk distribusi nol MI beserta nilai MI teramati.

    Ini adalah mesin di balik :func:`permutation_test`. Fungsi terpisah agar
    notebook dapat menggambar histogram distribusi nol tanpa perlu menulis
    ulang loop permutasinya (aturan kerja no. 8).

    Args:
        x: Vektor prediktor.
        y: Vektor target yang akan diacak.
        n_permutations: Banyaknya permutasi.
        random_state: Seed generator permutasi.
        n_neighbors: Jumlah tetangga terdekat pada estimator k-NN.

    Returns:
        Tuple ``(mi_observed, null_values)``; ``null_values`` berisi
        ``n_permutations`` nilai MI di bawah hipotesis nol.

    Raises:
        MutualInformationError: Bila ``n_permutations`` kurang dari satu.
    """
    if n_permutations < 1:
        raise MutualInformationError(
            f"n_permutations harus >= 1, diterima {n_permutations}."
        )

    x_clean, y_clean = _as_clean_pair(x, y)

    # Seed estimator dibuat tetap agar variasi antar-permutasi murni berasal
    # dari pengacakan, bukan dari derau internal estimator.
    mi_observed = compute_mi(
        x_clean, y_clean, n_neighbors=n_neighbors, random_state=random_state
    )

    rng = np.random.default_rng(random_state)
    null_values = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = rng.permutation(y_clean)
        null_values[i] = compute_mi(
            x_clean, shuffled, n_neighbors=n_neighbors, random_state=random_state
        )

    return mi_observed, null_values


def _permutation_test_detail(
    x: Any,
    y: Any,
    *,
    n_permutations: int,
    random_state: int,
    n_neighbors: int,
) -> dict[str, Any]:
    """Menjalankan uji permutasi dan mengembalikan rincian distribusi nol.

    Dipakai internal oleh :func:`permutation_test` (yang hanya mengembalikan
    nilai p) dan oleh pelaporan JSON yang membutuhkan ringkasan distribusi nol.

    Args:
        x: Vektor prediktor.
        y: Vektor target yang akan diacak.
        n_permutations: Banyaknya permutasi.
        random_state: Seed generator permutasi.
        n_neighbors: Jumlah tetangga terdekat pada estimator k-NN.

    Returns:
        Dictionary berisi MI teramati, ringkasan distribusi nol, dan nilai p.
    """
    mi_observed, null_values = permutation_null_distribution(
        x,
        y,
        n_permutations=n_permutations,
        random_state=random_state,
        n_neighbors=n_neighbors,
    )

    n_ge = int(np.count_nonzero(null_values >= mi_observed))

    return {
        "mi_observed": float(mi_observed),
        "n_permutations": int(n_permutations),
        "n_null_ge_observed": n_ge,
        "p_value_proportion": float(n_ge / n_permutations),
        "p_value": float((1 + n_ge) / (1 + n_permutations)),
        "null_mean": float(null_values.mean()),
        "null_std": float(null_values.std(ddof=1)),
        "null_q95": float(np.quantile(null_values, 0.95)),
        "null_max": float(null_values.max()),
    }


def compute_linear_correlations(x: Any, y: Any) -> dict[str, Any]:
    """Menghitung korelasi Pearson dan Spearman sebagai pembanding konteks MI.

    Pearson hanya menangkap hubungan linear, Spearman menangkap hubungan monoton,
    sedangkan MI menangkap hubungan bentuk apa pun. Perbandingan ketiganya
    membantu menjelaskan apakah keterkaitan kovariat bersifat linear atau tidak.

    Args:
        x: Vektor pertama.
        y: Vektor kedua.

    Returns:
        Dictionary ``{"pearson": {...}, "spearman": {...}}`` berisi koefisien
        dan nilai p masing-masing.
    """
    x_clean, y_clean = _as_clean_pair(x, y)

    pearson = stats.pearsonr(x_clean, y_clean)
    spearman = stats.spearmanr(x_clean, y_clean)

    return {
        "pearson": {
            "coefficient": float(pearson.statistic),
            "p_value": float(pearson.pvalue),
        },
        "spearman": {
            "coefficient": float(spearman.statistic),
            "p_value": float(spearman.pvalue),
        },
    }


# =============================================================================
# Transformasi data
# =============================================================================


def normalize_mode(mode: str) -> str:
    """Menormalkan nama varian menjadi salah satu nama kanonis.

    Args:
        mode: Nama varian, mis. ``"level"``, ``"logreturn"``, ``"log_return"``.

    Returns:
        Nama kanonis: ``"level"`` atau ``"log_return"``.

    Raises:
        MutualInformationError: Bila nama varian tidak dikenali.
    """
    key = str(mode).strip().lower()
    if key not in _MODE_ALIASES:
        raise MutualInformationError(
            f"Varian '{mode}' tidak dikenali. Pilihan: "
            f"{sorted(set(_MODE_ALIASES.values()))}."
        )
    return _MODE_ALIASES[key]


def transform_frame(df: pd.DataFrame, mode: str, columns: list[str]) -> pd.DataFrame:
    """Menyiapkan dataframe sesuai varian yang diminta.

    * ``level``      : kolom harga dikembalikan apa adanya.
    * ``log_return`` : tiap kolom diubah menjadi ``log(P_t / P_{t-1})`` dan
      baris pertama (yang selalu NaN) dibuang.

    Args:
        df: Dataframe data latih dengan kolom harga.
        mode: Nama varian (akan dinormalkan lebih dulu).
        columns: Kolom yang akan diambil/ditransformasi.

    Returns:
        Dataframe hasil transformasi, hanya berisi ``columns``.

    Raises:
        MutualInformationError: Bila ada kolom yang tidak tersedia atau ada
            harga non-positif saat menghitung log-return.
    """
    canonical_mode = normalize_mode(mode)

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise MutualInformationError(f"Kolom berikut tidak ada pada data: {missing}")

    subset = df.loc[:, columns].astype(float)

    if canonical_mode == MODE_LEVEL:
        return subset

    if (subset <= 0).to_numpy().any():
        raise MutualInformationError(
            "Terdapat harga <= 0, log-return tidak terdefinisi."
        )

    # log(P_t / P_{t-1}) == log(P_t) - log(P_{t-1}); baris pertama dibuang.
    return np.log(subset).diff().iloc[1:]


# =============================================================================
# Perhitungan tingkat dataset
# =============================================================================


def _mi_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Mengambil blok konfigurasi ``mutual_information`` beserta nilai turunannya.

    Args:
        config: Konfigurasi project.

    Returns:
        Dictionary parameter MI yang siap dipakai.
    """
    settings = dict(config["mutual_information"])
    settings["variants"] = [normalize_mode(v) for v in settings["variants"]]
    return settings


def compute_mi_matrix(
    df: pd.DataFrame,
    mode: str,
    config: dict[str, Any] | None = None,
    *,
    run_permutation_test: bool = True,
) -> dict[str, Any]:
    """Menghitung MI antara target (perak) dan tiap kovariat pada satu varian.

    Untuk tiap pasangan (perak, kovariat) dilaporkan MI stabil (rata-rata +/-
    simpangan baku dari beberapa pengulangan), korelasi Pearson dan Spearman
    sebagai pembanding, serta hasil uji permutasi.

    Args:
        df: Dataframe data latih (berisi kolom target dan kovariat).
        mode: ``"level"`` atau ``"log_return"``.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        run_permutation_test: Bila ``False``, uji permutasi dilewati (berguna
            untuk pemeriksaan cepat).

    Returns:
        Dictionary hasil untuk varian tersebut.
    """
    if config is None:
        config = load_config()

    canonical_mode = normalize_mode(mode)
    settings = _mi_settings(config)

    target_column = get_target_column(config)
    covariate_columns = get_covariate_columns(config)

    transformed = transform_frame(
        df, canonical_mode, [target_column, *covariate_columns]
    ).dropna()

    target_values = transformed[target_column].to_numpy()

    pairs: dict[str, Any] = {}
    for covariate in covariate_columns:
        covariate_values = transformed[covariate].to_numpy()

        mi_mean, mi_std = compute_mi_stable(
            covariate_values,
            target_values,
            n_repeats=settings["n_repeats"],
            n_neighbors=settings["n_neighbors"],
            random_state=settings["random_state"],
        )

        entry: dict[str, Any] = {
            "mi_mean": mi_mean,
            "mi_std": mi_std,
            "n_repeats": int(settings["n_repeats"]),
            **compute_linear_correlations(covariate_values, target_values),
        }

        if run_permutation_test:
            entry["permutation_test"] = _permutation_test_detail(
                covariate_values,
                target_values,
                n_permutations=settings["n_permutations"],
                random_state=settings["random_state"],
                n_neighbors=settings["n_neighbors"],
            )

        pairs[covariate] = entry

    return {
        "mode": canonical_mode,
        "mode_label": MODE_LABELS[canonical_mode],
        "target": target_column,
        "n_samples": int(len(transformed)),
        "pairs": pairs,
    }


def compute_mi_lagged(
    df: pd.DataFrame,
    lags: list[int],
    mode: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Menghitung MI antara ``silver_t`` dan ``kovariat_{t-lag}`` untuk tiap lag.

    Analisis ini menunjukkan apakah kovariat memiliki efek *lead* terhadap
    perak: bila MI pada lag > 0 tetap tinggi, nilai kovariat di masa lalu masih
    membawa informasi tentang perak hari ini — persis situasi yang dihadapi
    model saat kovariat diperlakukan sebagai *past-only* (keputusan D2).

    Args:
        df: Dataframe data latih.
        lags: Daftar lag (dalam hari perdagangan) yang diuji.
        mode: ``"level"`` atau ``"log_return"``.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.

    Returns:
        Dictionary ``{"mode": ..., "lags": [...], "pairs": {kovariat: [...]}}``
        dengan satu entri MI per lag.

    Raises:
        MutualInformationError: Bila ada lag negatif (melihat masa depan).
    """
    if config is None:
        config = load_config()

    canonical_mode = normalize_mode(mode)
    settings = _mi_settings(config)

    negative_lags = [lag for lag in lags if lag < 0]
    if negative_lags:
        raise MutualInformationError(
            f"Lag negatif tidak diperbolehkan (mengintip masa depan): {negative_lags}"
        )

    target_column = get_target_column(config)
    covariate_columns = get_covariate_columns(config)

    transformed = transform_frame(df, canonical_mode, [target_column, *covariate_columns])

    pairs: dict[str, list[dict[str, Any]]] = {}
    for covariate in covariate_columns:
        entries: list[dict[str, Any]] = []
        for lag in lags:
            # shift(lag) menempatkan kovariat_{t-lag} pada baris bertanggal t
            paired = pd.concat(
                [transformed[target_column], transformed[covariate].shift(lag)],
                axis=1,
                keys=["target", "covariate"],
            ).dropna()

            mi_mean, mi_std = compute_mi_stable(
                paired["covariate"].to_numpy(),
                paired["target"].to_numpy(),
                n_repeats=settings["n_repeats"],
                n_neighbors=settings["n_neighbors"],
                random_state=settings["random_state"],
            )

            entries.append(
                {
                    "lag": int(lag),
                    "n_samples": int(len(paired)),
                    "mi_mean": mi_mean,
                    "mi_std": mi_std,
                }
            )
        pairs[covariate] = entries

    return {
        "mode": canonical_mode,
        "mode_label": MODE_LABELS[canonical_mode],
        "target": target_column,
        "lags": [int(lag) for lag in lags],
        "pairs": pairs,
    }


# =============================================================================
# Visualisasi
# =============================================================================


def _style_axes(ax: plt.Axes) -> None:
    """Menerapkan gaya sumbu yang seragam: grid tipis, spine minimal.

    Args:
        ax: Objek axes matplotlib yang akan digaya.
    """
    ax.set_facecolor(_SURFACE)
    ax.grid(axis="y", color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=_INK_MUTED, labelsize=9, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(_INK_SECONDARY)


def _short_label(column: str) -> str:
    """Membuat label pendek untuk gambar dari nama kolom.

    Args:
        column: Nama kolom, mis. ``"gold_close"``.

    Returns:
        Label ringkas, mis. ``"GOLD"``.
    """
    return column.replace("_close", "").upper()


def plot_mi_barplot(
    results: dict[str, dict[str, Any]],
    output_path: str | Path,
    dpi: int,
) -> Path:
    """Menggambar MI level dan MI log-return berdampingan.

    Kedua varian ditempatkan pada panel terpisah dengan skala sumbu-y
    masing-masing, karena besaran MI level dan MI log-return berbeda beberapa
    kali lipat sehingga tidak layak dipaksakan pada satu sumbu.

    Args:
        results: Dictionary ``{mode: hasil compute_mi_matrix}``.
        output_path: Lokasi berkas gambar keluaran.
        dpi: Resolusi gambar.

    Returns:
        Path absolut berkas gambar yang tersimpan.
    """
    modes = [mode for mode in (MODE_LEVEL, MODE_LOG_RETURN) if mode in results]

    figure, axes = plt.subplots(
        1, len(modes), figsize=(9.5, 4.2), facecolor=_SURFACE
    )
    axes = np.atleast_1d(axes)

    for ax, mode in zip(axes, modes):
        entry = results[mode]
        covariates = list(entry["pairs"].keys())
        means = [entry["pairs"][c]["mi_mean"] for c in covariates]
        stds = [entry["pairs"][c]["mi_std"] for c in covariates]
        positions = np.arange(len(covariates))

        ax.bar(
            positions,
            means,
            width=0.55,
            color=list(_SERIES_COLORS[: len(covariates)]),
            yerr=stds,
            capsize=4,
            error_kw={"ecolor": _INK_SECONDARY, "elinewidth": 1.2},
            zorder=2,
        )

        # Label nilai langsung di atas tiap batang (identitas tidak hanya warna)
        headroom = max(means) * 0.14 if max(means) > 0 else 0.1
        for position, mean, std in zip(positions, means, stds):
            ax.text(
                position,
                mean + std + headroom * 0.25,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=_INK_PRIMARY,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels([_short_label(c) for c in covariates], fontsize=10)
        ax.set_ylim(0, max(means) + max(stds) + headroom)
        ax.set_title(
            f"{entry['mode_label']}  (n = {entry['n_samples']})",
            fontsize=11,
            color=_INK_PRIMARY,
            pad=10,
        )
        _style_axes(ax)

    axes[0].set_ylabel("Mutual Information (nat)", fontsize=10, color=_INK_SECONDARY)

    figure.suptitle(
        "Mutual Information perak terhadap kovariat — data latih",
        fontsize=12.5,
        color=_INK_PRIMARY,
        y=0.99,
    )
    figure.text(
        0.5,
        0.015,
        "Batang galat = simpangan baku antar-pengulangan estimator k-NN. "
        "Perhatikan skala sumbu-y kedua panel berbeda.",
        ha="center",
        fontsize=8.5,
        color=_INK_MUTED,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))

    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor=_SURFACE)
    plt.close(figure)

    return path


def _staggered_labels(
    endpoints: list[tuple[float, str]],
    ylim: tuple[float, float],
    min_gap_fraction: float = 0.07,
) -> list[tuple[float, str]]:
    """Menggeser posisi vertikal label agar tidak saling menimpa.

    Label ditempatkan pada nilai aslinya selama jaraknya cukup; bila terlalu
    rapat, label yang lebih atas didorong naik sejauh jarak minimum.

    Args:
        endpoints: Daftar ``(nilai_y, teks_label)``.
        ylim: Batas sumbu-y ``(bawah, atas)`` yang sedang berlaku.
        min_gap_fraction: Jarak minimum antar-label sebagai fraksi tinggi sumbu.

    Returns:
        Daftar ``(nilai_y_terkoreksi, teks_label)``.
    """
    min_gap = (ylim[1] - ylim[0]) * min_gap_fraction
    adjusted: list[tuple[float, str]] = []

    for value, text in sorted(endpoints):
        if adjusted and value - adjusted[-1][0] < min_gap:
            value = adjusted[-1][0] + min_gap
        adjusted.append((value, text))

    return adjusted


def plot_mi_lag(
    lagged: dict[str, dict[str, Any]],
    output_path: str | Path,
    dpi: int,
) -> Path:
    """Menggambar MI sebagai fungsi lag untuk kedua varian.

    Args:
        lagged: Dictionary ``{mode: hasil compute_mi_lagged}``.
        output_path: Lokasi berkas gambar keluaran.
        dpi: Resolusi gambar.

    Returns:
        Path absolut berkas gambar yang tersimpan.
    """
    modes = [mode for mode in (MODE_LEVEL, MODE_LOG_RETURN) if mode in lagged]

    figure, axes = plt.subplots(1, len(modes), figsize=(9.5, 4.2), facecolor=_SURFACE)
    axes = np.atleast_1d(axes)

    for ax, mode in zip(axes, modes):
        entry = lagged[mode]
        lags = entry["lags"]
        endpoints: list[tuple[float, str]] = []

        for color, (covariate, series) in zip(_SERIES_COLORS, entry["pairs"].items()):
            means = np.array([point["mi_mean"] for point in series], dtype=float)
            stds = np.array([point["mi_std"] for point in series], dtype=float)

            ax.fill_between(
                lags, means - stds, means + stds, color=color, alpha=0.15, linewidth=0
            )
            ax.plot(
                lags,
                means,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=6,
                markeredgecolor=_SURFACE,
                markeredgewidth=1.5,
                label=_short_label(covariate),
                zorder=3,
            )
            endpoints.append((float(means[-1]), _short_label(covariate)))

        ax.set_xticks(lags)
        ax.set_xlabel("Lag (hari perdagangan)", fontsize=10, color=_INK_SECONDARY)
        ax.set_xlim(min(lags) - 0.3, max(lags) + 1.1)
        ax.set_ylim(bottom=0)
        ax.set_title(entry["mode_label"], fontsize=11, color=_INK_PRIMARY, pad=10)
        _style_axes(ax)

        # Label langsung pada titik lag terakhir, digeser vertikal bila berimpit
        # (pada log-return seluruh seri jatuh ke ~0 sehingga labelnya bertumpuk)
        for label_y, label_text in _staggered_labels(endpoints, ax.get_ylim()):
            ax.annotate(
                label_text,
                xy=(lags[-1], label_y),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=9.5,
                color=_INK_SECONDARY,
            )

    axes[0].set_ylabel("Mutual Information (nat)", fontsize=10, color=_INK_SECONDARY)
    legend = axes[0].legend(
        frameon=False, fontsize=9.5, loc="best", labelcolor=_INK_SECONDARY
    )
    legend.set_title(None)

    figure.suptitle(
        "MI antara perak$_t$ dan kovariat$_{t-lag}$ — data latih",
        fontsize=12.5,
        color=_INK_PRIMARY,
        y=0.99,
    )
    figure.text(
        0.5,
        0.015,
        "Pita = rata-rata +/- simpangan baku antar-pengulangan. "
        "Perhatikan skala sumbu-y kedua panel berbeda.",
        ha="center",
        fontsize=8.5,
        color=_INK_MUTED,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))

    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor=_SURFACE)
    plt.close(figure)

    return path


# =============================================================================
# Pelaporan
# =============================================================================


def build_markdown_tables(
    results: dict[str, dict[str, Any]], lagged: dict[str, dict[str, Any]]
) -> str:
    """Menyusun tabel ringkas format markdown untuk disalin ke naskah skripsi.

    Args:
        results: Dictionary ``{mode: hasil compute_mi_matrix}``.
        lagged: Dictionary ``{mode: hasil compute_mi_lagged}``.

    Returns:
        String multi-baris berisi dua tabel markdown.
    """
    lines: list[str] = []

    lines.append("### Tabel 1. Mutual Information dan korelasi perak vs kovariat "
                 "(data latih)")
    lines.append("")
    lines.append(
        "| Varian | Kovariat | n | MI (nat) | SD MI | Pearson r | Spearman rho | "
        "p (permutasi) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for mode in (MODE_LEVEL, MODE_LOG_RETURN):
        if mode not in results:
            continue
        entry = results[mode]
        for covariate, pair in entry["pairs"].items():
            permutation = pair.get("permutation_test", {})
            p_value = permutation.get("p_value")
            p_text = "-" if p_value is None else f"{p_value:.4f}"
            lines.append(
                f"| {entry['mode_label']} | {_short_label(covariate)} | "
                f"{entry['n_samples']} | {pair['mi_mean']:.4f} | {pair['mi_std']:.4f} | "
                f"{pair['pearson']['coefficient']:+.4f} | "
                f"{pair['spearman']['coefficient']:+.4f} | {p_text} |"
            )

    lines.append("")
    lines.append("### Tabel 2. Mutual Information menurut lag kovariat (data latih)")
    lines.append("")

    reference_mode = MODE_LEVEL if MODE_LEVEL in lagged else next(iter(lagged))
    lags = lagged[reference_mode]["lags"]

    header = "| Varian | Kovariat | " + " | ".join(f"lag {lag}" for lag in lags) + " |"
    separator = "|---|---|" + "|".join(["---:"] * len(lags)) + "|"
    lines.append(header)
    lines.append(separator)

    for mode in (MODE_LEVEL, MODE_LOG_RETURN):
        if mode not in lagged:
            continue
        entry = lagged[mode]
        for covariate, series in entry["pairs"].items():
            cells = " | ".join(f"{point['mi_mean']:.4f}" for point in series)
            lines.append(
                f"| {entry['mode_label']} | {_short_label(covariate)} | {cells} |"
            )

    lines.append("")
    lines.append("*MI dalam satuan nat; seluruh angka dihitung hanya dari data latih "
                 "(keputusan D6).*")

    return "\n".join(lines)


def build_interpretation(
    results: dict[str, dict[str, Any]], lagged: dict[str, dict[str, Any]]
) -> str:
    """Menyusun paragraf interpretasi otomatis berdasarkan angka yang terhitung.

    Args:
        results: Dictionary ``{mode: hasil compute_mi_matrix}``.
        lagged: Dictionary ``{mode: hasil compute_mi_lagged}``.

    Returns:
        Paragraf interpretasi siap salin ke naskah.
    """
    level = results[MODE_LEVEL]["pairs"]
    log_return = results[MODE_LOG_RETURN]["pairs"]

    def _describe(pairs: dict[str, Any]) -> str:
        return " dan ".join(
            f"{_short_label(name)} = {pair['mi_mean']:.3f}" for name, pair in pairs.items()
        )

    # Rasio penyusutan MI level -> log-return, dihitung per kovariat
    ratios = [
        level[name]["mi_mean"] / log_return[name]["mi_mean"]
        for name in level
        if log_return.get(name, {}).get("mi_mean", 0.0) > 0
    ]
    ratio_text = (
        f"{min(ratios):.1f} sampai {max(ratios):.1f} kali lebih kecil"
        if ratios
        else "jauh lebih kecil"
    )

    # Signifikansi diambil dari varian log-return (varian yang asumsinya sah)
    p_values = [
        pair.get("permutation_test", {}).get("p_value", 1.0)
        for pair in log_return.values()
    ]
    all_significant = all(p < 0.05 for p in p_values)
    verdict = (
        f"kedua kovariat layak dipakai: uji permutasi pada log-return menolak "
        f"hipotesis kemandirian untuk keduanya (p <= {max(p_values):.3f})"
        if all_significant
        else "kelayakan kovariat perlu dibaca hati-hati karena tidak semua pasangan "
        "signifikan pada varian log-return"
    )

    # Kontras MI vs korelasi linear: kovariat dengan |Pearson| terkecil pada level
    weakest_linear = min(
        level.items(), key=lambda item: abs(item[1]["pearson"]["coefficient"])
    )
    contrast_text = (
        f"{_short_label(weakest_linear[0])} bahkan menyimpan MI "
        f"{weakest_linear[1]['mi_mean']:.3f} nat pada level padahal korelasi "
        f"Pearson-nya hanya {weakest_linear[1]['pearson']['coefficient']:+.3f}, "
        f"pertanda keterkaitannya tidak tertangkap ukuran linear"
    )

    # Perbandingan MI pada lag 0 versus lag > 0 pada varian log-return
    lag_series = lagged[MODE_LOG_RETURN]["pairs"]
    mi_at_zero = {
        name: next(p["mi_mean"] for p in series if p["lag"] == 0)
        for name, series in lag_series.items()
    }
    mi_beyond_zero = [
        p["mi_mean"] for series in lag_series.values() for p in series if p["lag"] > 0
    ]
    max_beyond_zero = max(mi_beyond_zero) if mi_beyond_zero else 0.0
    max_at_zero = max(mi_at_zero.values())
    lead_effect_absent = max_beyond_zero < 0.1 * max_at_zero

    lag_text = (
        f"MI log-return runtuh menjadi paling besar {max_beyond_zero:.3f} nat begitu "
        f"lag >= 1 (dari {max_at_zero:.3f} nat pada lag 0), sehingga kovariat bersifat "
        f"sezaman dan bukan penunjuk arah (lead) bagi perak"
        if lead_effect_absent
        else f"MI log-return masih mencapai {max_beyond_zero:.3f} nat pada lag >= 1 "
        f"(dibanding {max_at_zero:.3f} nat pada lag 0), sehingga kovariat menyimpan "
        f"efek lead yang bertahan beberapa hari"
    )

    return (
        f"Pada level harga MI perak terhadap kovariat mencapai {_describe(level)} nat, "
        f"sedangkan pada log-return turun menjadi {_describe(log_return)} nat — yakni "
        f"{ratio_text}. "
        f"Meski begitu {verdict}, dan {contrast_text}. "
        f"Di sisi lain {lag_text}; manfaatnya bagi Chronos-2 karena itu terletak pada "
        f"pengayaan konteks past-only (keputusan D2), bukan pada kemampuan meramal "
        f"beberapa hari ke depan dari kovariat saja. "
        f"MI level jauh lebih besar karena harga perak, emas, dan dolar sama-sama "
        f"merupakan proses non-stasioner yang berbagi tren jangka panjang: pada level "
        f"estimator k-NN sesungguhnya mengukur kesamaan lintasan tren, bukan "
        f"keterkaitan pergerakan harian, sehingga MI level melebih-lebihkan kandungan "
        f"informasi yang benar-benar dapat dieksploitasi untuk peramalan."
    )


# =============================================================================
# Pemuatan data & titik masuk
# =============================================================================


def _relative_to_project(path: Path) -> str:
    """Mengubah path absolut menjadi path relatif terhadap akar project.

    Dipakai agar berkas hasil tetap portabel antar-mesin.

    Args:
        path: Path absolut berkas keluaran.

    Returns:
        Path relatif dengan pemisah ``/``; path di luar akar project
        dikembalikan apa adanya.
    """
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_train_data(
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Memuat data latih dari ``data/processed/train.csv``.

    Hanya berkas latih yang dibaca. Data validasi dan data uji sengaja tidak
    disentuh agar analisis kovariat tidak menimbulkan kebocoran data.

    Args:
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dataframe data latih dengan index tanggal.

    Raises:
        FileNotFoundError: Bila berkas latih belum ada.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("mutual_information")

    path = resolve_path(config["data"]["processed_dir"]) / "train.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Berkas data latih '{path}' tidak ditemukan. "
            f"Jalankan 'python -m src.preprocessing' terlebih dahulu."
        )

    frame = pd.read_csv(path, index_col=DATE_INDEX_NAME, parse_dates=[DATE_INDEX_NAME])
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame.index.name = DATE_INDEX_NAME
    frame = frame.sort_index()

    logger.info(
        "Memuat data latih: %d baris (%s s.d. %s) dari %s",
        len(frame),
        frame.index[0].date(),
        frame.index[-1].date(),
        path,
    )
    logger.info("Data validasi dan data uji TIDAK dibaca (keputusan D6 / aturan no. 2).")

    return frame


def main() -> dict[str, Any]:
    """Titik masuk eksekusi mandiri modul.

    Menghitung MI (level dan log-return), MI berlag, korelasi pembanding, dan
    uji permutasi dari data latih; menyimpan hasil ke JSON dan dua gambar; lalu
    mencetak tabel markdown serta interpretasi ke stdout.

    Returns:
        Dictionary laporan lengkap yang juga disimpan ke JSON.
    """
    config = load_config()
    logger = setup_logger(
        "mutual_information", log_dir=config.get("paths", {}).get("logs_dir", "logs")
    )

    settings = _mi_settings(config)
    target_column = get_target_column(config)
    covariate_columns = get_covariate_columns(config)

    logger.info("=" * 78)
    logger.info("MUTUAL INFORMATION (keputusan D6) — hanya memakai data latih")
    logger.info("=" * 78)
    logger.info(
        "Parameter: n_neighbors=%d, n_repeats=%d, n_permutations=%d, random_state=%d",
        settings["n_neighbors"],
        settings["n_repeats"],
        settings["n_permutations"],
        settings["random_state"],
    )

    train = load_train_data(config=config, logger=logger)

    results: dict[str, dict[str, Any]] = {}
    lagged: dict[str, dict[str, Any]] = {}

    for mode in settings["variants"]:
        logger.info("-" * 78)
        logger.info("Varian '%s' — MI, korelasi, dan uji permutasi", mode)
        results[mode] = compute_mi_matrix(train, mode, config=config)
        for covariate, pair in results[mode]["pairs"].items():
            logger.info(
                "  %-12s MI = %.4f +/- %.4f | Pearson = %+.4f | Spearman = %+.4f | "
                "p_perm = %.4f",
                covariate,
                pair["mi_mean"],
                pair["mi_std"],
                pair["pearson"]["coefficient"],
                pair["spearman"]["coefficient"],
                pair["permutation_test"]["p_value"],
            )

        logger.info("Varian '%s' — MI menurut lag %s", mode, settings["lags"])
        lagged[mode] = compute_mi_lagged(
            train, settings["lags"], mode, config=config
        )
        for covariate, series in lagged[mode]["pairs"].items():
            logger.info(
                "  %-12s %s",
                covariate,
                ", ".join(f"lag {p['lag']}: {p['mi_mean']:.4f}" for p in series),
            )

    figures_dir = config["paths"]["figures_dir"]
    barplot_path = plot_mi_barplot(
        results, Path(figures_dir) / "mi_barplot.png", dpi=settings["figure_dpi"]
    )
    lag_plot_path = plot_mi_lag(
        lagged, Path(figures_dir) / "mi_lag_plot.png", dpi=settings["figure_dpi"]
    )
    logger.info("Gambar tersimpan: %s", barplot_path)
    logger.info("Gambar tersimpan: %s", lag_plot_path)

    interpretation = build_interpretation(results, lagged)

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "D6 — MI dihitung hanya dari data latih",
        "settings": {
            "n_neighbors": int(settings["n_neighbors"]),
            "n_repeats": int(settings["n_repeats"]),
            "n_permutations": int(settings["n_permutations"]),
            "random_state": int(settings["random_state"]),
            "lags": [int(lag) for lag in settings["lags"]],
            "variants": settings["variants"],
            "estimator": "sklearn.feature_selection.mutual_info_regression (k-NN)",
            "unit": "nat",
        },
        "data": {
            "source": "data/processed/train.csv",
            "n_rows": int(len(train)),
            "date_start": str(train.index[0].date()),
            "date_end": str(train.index[-1].date()),
            "target": target_column,
            "covariates": covariate_columns,
        },
        "results": results,
        "lagged": lagged,
        "figures": {
            "mi_barplot": _relative_to_project(barplot_path),
            "mi_lag_plot": _relative_to_project(lag_plot_path),
        },
        "notes": [
            "MI dilaporkan sebagai rata-rata +/- simpangan baku dari n_repeats "
            "pengulangan karena estimator k-NN bersifat stokastik.",
            "Nilai p permutasi memakai koreksi tambah-satu: (1 + jumlah MI_null >= "
            "MI_observed) / (1 + n_permutations).",
            "Pada varian level, uji permutasi bersifat anti-konservatif karena harga "
            "berautokorelasi kuat sehingga asumsi pengamatan saling bebas dilanggar; "
            "varian log-return jauh lebih mendekati asumsi tersebut.",
            "Lag berarti MI antara target_t dan kovariat_{t-lag}; lag negatif "
            "ditolak karena akan mengintip masa depan.",
        ],
        "interpretation": interpretation,
    }

    report_path = save_json(
        report, resolve_path(config["paths"]["metrics_dir"]) / "mutual_information.json"
    )
    logger.info("Hasil lengkap tersimpan di %s", report_path)

    tables = build_markdown_tables(results, lagged)
    print()
    print(tables)
    print()
    print("### Interpretasi")
    print()
    print(interpretation)
    print()

    return report


if __name__ == "__main__":
    main()
