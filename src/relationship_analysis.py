"""Analisis hubungan antar-variabel: Perak, Emas, dan Dollar Index.

Ini adalah tujuan penelitian **pertama**, setara kedudukannya dengan peramalan
probabilistik (bukan tahap pendukung untuk memilih kovariat). Modul ini
menjawab pertanyaan: apakah ketiga seri bergerak bersama, pada lag berapa, dan
ke arah mana presedensi prediktifnya — bukan mencari sinyal beli/jual maupun
menyaring kovariat mana yang "layak" dipakai model.

Seluruh perhitungan memakai **data latih saja** (``data/processed/train.csv``).
Data validasi dan data uji tidak pernah disentuh di modul ini.

Urutan wajib, karena tiap langkah menentukan validitas langkah berikutnya:

1. :func:`test_stationarity`      — ADF & KPSS pada level dan log-return.
   Granger pada seri non-stasioner menghasilkan regresi lancung.
2. :func:`plot_acf_pacf`          — ACF/PACF log-return ketiga seri.
3. :func:`cross_correlation`      — CCF perak vs tiap kovariat.
4. :func:`test_cointegration`     — Engle-Granger & Johansen pada level.
   Perak-Emas berpotensi terkointegrasi (rasio emas-perak dikenal luas).
5. :func:`select_var_lag`         — pemilihan lag VAR via AIC/BIC/HQIC/FPE.
6. :func:`diagnose_var_residuals` — Ljung-Box & ARCH-LM pada residual VAR di
   lag terpilih (BIC). Bila Ljung-Box menolak H0, lag dinaikkan dan model
   diestimasi ulang (:func:`fit_var_with_ljungbox_escalation`) sampai lolos
   atau batas lag tercapai -- spesifikasi yang residualnya masih berautokorelasi
   tidak valid dipakai untuk langkah berikutnya.
7. :func:`granger_toda_yamamoto`  — presedensi prediktif dua arah pada lag
   hasil langkah 6, robust terhadap ada/tidaknya kointegrasi dari langkah 4
   (karena itu prosedurnya dipilih, bukan uji Granger standar pada data yang
   mungkin sudah dibedakan).
8. Mutual Information (dari :mod:`src.mutual_information`) sebagai pelengkap
   non-linear — Granger bersifat linear, MI menangkap ketergantungan bentuk
   apa pun. Ketidaksesuaian antara keduanya (mis. MI tinggi tapi Granger tidak
   signifikan) adalah temuan yang dilaporkan, bukan kegagalan yang "diperbaiki"
   dengan mengubah-ubah parameter sampai signifikan.

Istilah terlarang
    Granger causality bukan causal inference. Seluruh teks interpretasi hasil
    Granger WAJIB memakai "presedensi prediktif" atau "X Granger-cause Y", dan
    DILARANG memakai kata bersifat sebab-akibat ("menyebabkan", "pengaruh",
    "dampak", "efek"). Fungsi :func:`assert_no_causal_language` memindai setiap
    teks interpretasi yang dihasilkan modul ini sebagai pengaman otomatis.

Cara menjalankan mandiri:
    python -m src.relationship_analysis

Keluaran:
    results/metrics/relationship_analysis.json
    results/metrics/relationship_summary.md
    results/figures/relationship/acf_pacf.png
    results/figures/relationship/ccf_perak_emas.png
    results/figures/relationship/ccf_perak_dxy.png
"""

from __future__ import annotations

import logging
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

# Backend non-interaktif: modul harus bisa jalan tanpa display (mis. di server)
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: E402
from statsmodels.regression.linear_model import OLS  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch  # noqa: E402
from statsmodels.tools import add_constant  # noqa: E402
from statsmodels.tools.sm_exceptions import InterpolationWarning, ValueWarning  # noqa: E402
from statsmodels.tsa.stattools import adfuller, coint, kpss  # noqa: E402
from statsmodels.tsa.vector_ar.var_model import VAR, VARResultsWrapper  # noqa: E402
from statsmodels.tsa.vector_ar.vecm import coint_johansen  # noqa: E402

from src.data_collection import DATE_INDEX_NAME  # noqa: E402
from src.preprocessing import get_covariate_columns, get_target_column  # noqa: E402
from src.utils import (  # noqa: E402
    PROJECT_ROOT,
    load_config,
    resolve_path,
    save_json,
    setup_logger,
)


class RelationshipAnalysisError(ValueError):
    """Kegagalan pada tahap analisis hubungan antar-variabel."""


# --- Palet gambar (sama seperti src/mutual_information.py dan src/visualization.py,
# supaya seluruh gambar pada laporan tampak berasal dari satu sistem visual) --------
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#3f8f5f")  # biru, oranye, hijau


# =============================================================================
# Pengaman bahasa: Granger causality bukan causal inference
# =============================================================================

# Akar/kata yang menyiratkan sebab-akibat. Dipakai sebagai substring (bukan kata
# utuh) supaya konjugasi ikut tertangkap, mis. "sebab" menangkap "disebabkan",
# "tersebab", "sebab-akibat"; "pengaruh" menangkap "berpengaruh", "mempengaruhi".
# "menyebabkan" dan "penyebab" WAJIB dicantumkan eksplisit karena nasalisasi
# awalan meN-/peN- menghapus huruf 's' pada akar "sebab" (sebab -> nyebab),
# sehingga substring "sebab" tidak akan pernah cocok pada kedua kata itu.
# Sebab yang sama berlaku pada "memengaruhi" (varian baku "mempengaruhi") --
# nasalisasi meN- di sana menghapus huruf 'p' pada akar "pengaruh".
_FORBIDDEN_CAUSAL_ROOTS: tuple[str, ...] = (
    "sebab",
    "menyebabkan",
    "penyebab",
    "pengaruh",
    "memengaruhi",
    "dampak",
    "efek",
)

_FORBIDDEN_PATTERN = re.compile(
    "|".join(re.escape(root) for root in _FORBIDDEN_CAUSAL_ROOTS), re.IGNORECASE
)


def assert_no_causal_language(text: str) -> None:
    """Memindai teks keluaran dan menolak bila memakai kata bernuansa sebab-akibat.

    Hasil uji Granger causality (termasuk prosedur Toda-Yamamoto) tidak pernah
    boleh ditulis seolah membuktikan hubungan sebab-akibat. Fungsi ini adalah
    pengaman otomatis, bukan formalitas: setiap teks interpretasi yang dihasilkan
    modul ini WAJIB melewati pemeriksaan ini sebelum disimpan atau dicetak.

    Args:
        text: Teks yang akan diperiksa.

    Raises:
        RelationshipAnalysisError: Bila ditemukan kata terlarang. Pesan error
            menyertakan kata yang terdeteksi dan teks lengkapnya agar mudah
            ditelusuri dan diperbaiki.
    """
    match = _FORBIDDEN_PATTERN.search(text)
    if match:
        raise RelationshipAnalysisError(
            f"Teks keluaran memakai kata bernuansa sebab-akibat '{match.group(0)}' "
            "untuk hasil Granger causality. Granger causality bukan causal "
            "inference -- gunakan 'presedensi prediktif' atau 'X Granger-cause Y'. "
            f"Teks yang diperiksa: {text!r}"
        )


# =============================================================================
# 1. Stasioneritas
# =============================================================================


def _log_return(series: pd.Series) -> pd.Series:
    """Mengubah seri harga menjadi log-return, membuang baris pertama (NaN).

    Args:
        series: Seri harga (harus seluruhnya positif).

    Returns:
        Seri log-return: ``log(P_t / P_{t-1})``.

    Raises:
        RelationshipAnalysisError: Bila ada harga <= 0.
    """
    if (series <= 0).any():
        raise RelationshipAnalysisError(
            f"Seri '{series.name}' memuat nilai <= 0, log-return tidak terdefinisi."
        )
    return np.log(series).diff().dropna()


def _run_adf(series: pd.Series, alpha: float, regression: str) -> dict[str, Any]:
    """Menjalankan uji Augmented Dickey-Fuller (H0: seri memiliki unit root)."""
    stat, pvalue, used_lag, n_obs, crit_values, _ = adfuller(
        series.to_numpy(), regression=regression, autolag="AIC"
    )
    return {
        "statistic": float(stat),
        "p_value": float(pvalue),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "critical_values": {k: float(v) for k, v in crit_values.items()},
        # Tolak H0 (unit root) -> stasioner
        "stationary": bool(pvalue < alpha),
    }


def _run_kpss(series: pd.Series, alpha: float, regression: str) -> dict[str, Any]:
    """Menjalankan uji KPSS (H0: seri stasioner)."""
    with warnings.catch_warnings():
        # p-value KPSS statsmodels dibatasi tabel referensi (0.01 s.d. 0.1);
        # bila statistik uji ada di luar rentang itu, statsmodels memperingatkan
        # lewat InterpolationWarning tapi tetap mengembalikan batas p-value yang
        # valid (di-clip). Peringatan ini bukan indikasi kegagalan, jadi diredam.
        warnings.simplefilter("ignore", InterpolationWarning)
        stat, pvalue, n_lags, crit_values = kpss(
            series.to_numpy(), regression=regression, nlags="auto"
        )
    return {
        "statistic": float(stat),
        "p_value": float(pvalue),
        "n_lags": int(n_lags),
        "critical_values": {k: float(v) for k, v in crit_values.items()},
        # Gagal tolak H0 (stasioner) -> stasioner
        "stationary": bool(pvalue > alpha),
    }


def _combine_verdict(adf_result: dict[str, Any], kpss_result: dict[str, Any]) -> str:
    """Menggabungkan kesimpulan ADF dan KPSS; melaporkan konflik apa adanya."""
    if adf_result["stationary"] and kpss_result["stationary"]:
        return "stasioner"
    if not adf_result["stationary"] and not kpss_result["stationary"]:
        return "non_stasioner"
    return "tidak_konklusif"


def test_stationarity(
    series: pd.Series,
    name: str,
    alpha: float = 0.05,
    regression: str = "c",
) -> dict[str, Any]:
    """Menguji stasioneritas sebuah seri harga dengan ADF dan KPSS.

    Diuji pada dua transformasi:

    * ``level``      : harga apa adanya.
    * ``log_return`` : ``log(P_t / P_{t-1})``.

    ADF menguji H0 = ada unit root (non-stasioner); KPSS menguji H0 = seri
    stasioner. Keduanya dipakai berpasangan karena punya arah kesalahan yang
    berlawanan, sehingga bila sepakat kesimpulannya jauh lebih meyakinkan
    dibanding memakai satu uji saja. Bila kedua uji bertentangan pada satu
    transformasi, transformasi itu dilaporkan sebagai "tidak_konklusif" --
    tidak dipaksa memilih salah satu.

    Ordo integrasi ``d`` disimpulkan dari kombinasi kedua transformasi:
    ``d=0`` bila level sudah stasioner, ``d=1`` bila level non-stasioner/tidak
    konklusif namun log-return stasioner. Bila log-return masih non-stasioner
    atau tidak konklusif, ordo integrasi tidak dapat disimpulkan dari kedua
    transformasi yang diuji di sini (modul ini tidak menguji difference kedua).

    Args:
        series: Seri harga (level, bukan log-return) dengan index tanggal.
        name: Nama seri untuk label pada hasil.
        alpha: Taraf nyata uji.
        regression: Komponen deterministik ADF/KPSS (``"c"`` = konstanta saja).

    Returns:
        Dictionary berisi hasil ADF & KPSS pada level dan log-return, verdict
        gabungan tiap transformasi, dan kesimpulan ordo integrasi.
    """
    log_return = _log_return(series)

    level_adf = _run_adf(series, alpha, regression)
    level_kpss = _run_kpss(series, alpha, regression)
    level_verdict = _combine_verdict(level_adf, level_kpss)

    return_adf = _run_adf(log_return, alpha, regression)
    return_kpss = _run_kpss(log_return, alpha, regression)
    return_verdict = _combine_verdict(return_adf, return_kpss)

    if level_verdict == "stasioner":
        d: int | None = 0
        conclusive = True
        note = "Level sudah stasioner (ADF dan KPSS sepakat); d = 0."
    elif return_verdict == "stasioner":
        d = 1
        conclusive = level_verdict != "tidak_konklusif"
        note = (
            "Level non-stasioner dan log-return stasioner (ADF dan KPSS sepakat "
            "pada log-return); d = 1."
            if level_verdict == "non_stasioner"
            else "Level tidak konklusif, tetapi log-return stasioner (ADF dan KPSS "
            "sepakat); d = 1 diambil sebagai kesimpulan kerja."
        )
    else:
        d = None
        conclusive = False
        note = (
            f"Level berstatus '{level_verdict}' dan log-return berstatus "
            f"'{return_verdict}' -- ordo integrasi tidak dapat disimpulkan dari "
            "kedua transformasi yang diuji (level dan log-return orde pertama)."
        )

    return {
        "name": name,
        "alpha": alpha,
        "regression": regression,
        "level": {"adf": level_adf, "kpss": level_kpss, "verdict": level_verdict},
        "log_return": {"adf": return_adf, "kpss": return_kpss, "verdict": return_verdict},
        "order_of_integration": {"d": d, "conclusive": conclusive, "note": note},
    }


# =============================================================================
# 2. ACF dan PACF
# =============================================================================


def _style_axes(ax: plt.Axes) -> None:
    """Menerapkan gaya sumbu yang seragam: grid tipis, spine minimal."""
    ax.set_facecolor(_SURFACE)
    ax.grid(axis="y", color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=_INK_MUTED, labelsize=8.5, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(_INK_SECONDARY)


def _short_label(column: str) -> str:
    """Membuat label pendek untuk gambar dari nama kolom, mis. 'gold_close' -> 'GOLD'."""
    return column.replace("_close", "").upper()


def plot_acf_pacf(
    df: pd.DataFrame,
    mode: str = "logreturn",
    lags: int = 40,
    output_path: str | Path = "results/figures/relationship/acf_pacf.png",
    dpi: int = 200,
) -> Path:
    """Menggambar ACF dan PACF ketiga seri, dengan pita kepercayaan 95%.

    Args:
        df: Dataframe data latih berisi kolom harga ketiga seri.
        mode: ``"logreturn"`` (default, sesuai keputusan metodologis -- ACF/PACF
            level pada harga finansial didominasi tren dan tidak informatif)
            atau ``"level"``.
        lags: Jumlah lag yang ditampilkan.
        output_path: Lokasi berkas gambar keluaran.
        dpi: Resolusi gambar.

    Returns:
        Path absolut berkas gambar yang tersimpan.

    Raises:
        RelationshipAnalysisError: Bila ``mode`` tidak dikenali.
    """
    if mode not in ("logreturn", "level"):
        raise RelationshipAnalysisError(f"mode '{mode}' tidak dikenali (pilih 'logreturn'/'level').")

    columns = list(df.columns)
    if mode == "logreturn":
        plotted = pd.concat({col: _log_return(df[col]) for col in columns}, axis=1)
        mode_label = "Log-return"
    else:
        plotted = df.loc[:, columns]
        mode_label = "Level harga"

    figure, axes = plt.subplots(
        len(columns), 2, figsize=(9.5, 2.9 * len(columns)), facecolor=_SURFACE
    )

    for row, column in enumerate(columns):
        series = plotted[column].dropna()
        color = _SERIES_COLORS[row % len(_SERIES_COLORS)]

        plot_acf(
            series, ax=axes[row, 0], lags=lags, color=color, vlines_kwargs={"colors": color}
        )
        plot_pacf(
            series,
            ax=axes[row, 1],
            lags=lags,
            method="ywm",
            color=color,
            vlines_kwargs={"colors": color},
        )

        for ax in (axes[row, 0], axes[row, 1]):
            ax.set_title("")
            _style_axes(ax)
        axes[row, 0].set_ylabel(_short_label(column), fontsize=10, color=_INK_PRIMARY)

    axes[0, 0].set_title("ACF", fontsize=11, color=_INK_PRIMARY, pad=8)
    axes[0, 1].set_title("PACF", fontsize=11, color=_INK_PRIMARY, pad=8)

    figure.suptitle(
        f"ACF dan PACF -- {mode_label} (data latih)",
        fontsize=12.5,
        color=_INK_PRIMARY,
        y=0.995,
    )
    figure.text(
        0.5,
        0.008,
        "Pita biru muda = batas kepercayaan 95% di bawah hipotesis nol tanpa autokorelasi.",
        ha="center",
        fontsize=8.5,
        color=_INK_MUTED,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.97))

    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor=_SURFACE)
    plt.close(figure)

    return path


# =============================================================================
# 3. Cross-correlation function
# =============================================================================


def cross_correlation(x: pd.Series, y: pd.Series, max_lag: int = 10) -> pd.DataFrame:
    """Menghitung cross-correlation function (CCF) antara dua seri.

    Konvensi tanda lag (WAJIB dibaca sebelum menafsirkan hasil):
        ``CCF(lag) = corr(x_t, y_{t+lag})``. Untuk **lag positif**, nilai
        ``y`` yang dipasangkan berasal dari masa depan relatif terhadap ``x``
        -- artinya nilai ``x`` hari ini berkorelasi dengan nilai ``y`` di masa
        depan, sehingga **lag positif berarti x mendahului y** (x leads y).
        Sebaliknya, **lag negatif berarti y mendahului x**. Lag 0 adalah
        korelasi sezaman (kontemporer). Konvensi ini konsisten di seluruh
        modul -- membalik salah tafsirnya berarti membalik seluruh kesimpulan
        arah presedensi.

    Args:
        x: Seri pertama (kandidat "pemimpin" pada lag positif).
        y: Seri kedua (kandidat "pengikut" pada lag positif).
        max_lag: Rentang lag yang diuji, dari ``-max_lag`` sampai ``+max_lag``.

    Returns:
        DataFrame dengan kolom ``lag``, ``ccf``, ``n_obs``, ``signif_bound``
        (batas signifikansi aproksimasi ``2/sqrt(n)``), dan ``significant``.
    """
    aligned = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    n = len(aligned)
    signif_bound = 2.0 / np.sqrt(n)

    rows: list[dict[str, Any]] = []
    for lag in range(-max_lag, max_lag + 1):
        # y.shift(-lag) pada baris t bernilai y_{t+lag}
        shifted = pd.concat(
            [aligned["x"], aligned["y"].shift(-lag)], axis=1
        ).dropna()
        ccf = float(shifted.iloc[:, 0].corr(shifted.iloc[:, 1])) if len(shifted) > 1 else float("nan")
        rows.append(
            {
                "lag": lag,
                "ccf": ccf,
                "n_obs": int(len(shifted)),
                "signif_bound": signif_bound,
                "significant": bool(abs(ccf) > signif_bound) if np.isfinite(ccf) else False,
            }
        )

    return pd.DataFrame(rows)


def plot_cross_correlation(
    ccf_df: pd.DataFrame,
    x_label: str,
    y_label: str,
    output_path: str | Path,
    dpi: int = 200,
) -> Path:
    """Menggambar CCF sebagai stem plot dengan batas signifikansi.

    Args:
        ccf_df: Hasil :func:`cross_correlation`.
        x_label: Nama singkat seri ``x`` (mis. ``"EMAS"``), untuk judul/label.
        y_label: Nama singkat seri ``y`` (mis. ``"PERAK"``).
        output_path: Lokasi berkas gambar keluaran.
        dpi: Resolusi gambar.

    Returns:
        Path absolut berkas gambar yang tersimpan.
    """
    figure, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=_SURFACE)

    colors = [_SERIES_COLORS[0] if s else _BASELINE for s in ccf_df["significant"]]
    ax.bar(ccf_df["lag"], ccf_df["ccf"], color=colors, width=0.6, zorder=2)
    ax.axhline(0.0, color=_INK_MUTED, linewidth=0.8, zorder=1)

    bound = float(ccf_df["signif_bound"].iloc[0])
    ax.axhline(bound, color=_INK_SECONDARY, linewidth=1.0, linestyle="--", zorder=1)
    ax.axhline(-bound, color=_INK_SECONDARY, linewidth=1.0, linestyle="--", zorder=1)

    ax.set_xticks(ccf_df["lag"])
    ax.set_xlabel(
        f"lag (positif = {x_label} mendahului {y_label}; negatif = sebaliknya)",
        fontsize=9.5,
        color=_INK_SECONDARY,
    )
    ax.set_ylabel("Cross-correlation", fontsize=10, color=_INK_SECONDARY)
    ax.set_title(
        f"CCF {x_label} vs {y_label} -- log-return (data latih)",
        fontsize=12,
        color=_INK_PRIMARY,
        pad=10,
    )
    _style_axes(ax)

    figure.text(
        0.5,
        0.01,
        "Garis putus-putus = batas signifikansi aproksimasi +/- 2/sqrt(n).",
        ha="center",
        fontsize=8.5,
        color=_INK_MUTED,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))

    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor=_SURFACE)
    plt.close(figure)

    return path


# =============================================================================
# 4. Kointegrasi
# =============================================================================


def _select_aux_lag(df: pd.DataFrame, maxlags: int) -> int:
    """Memilih lag bantu (via AIC) untuk menurunkan ``k_ar_diff`` Johansen.

    Lag bantu ini independen dari lag utama yang dipilih pada
    :func:`select_var_lag` (langkah kointegrasi mendahului pemilihan lag pada
    urutan analisis). ``k_ar_diff`` VECM = (lag VAR pada level) - 1.

    Args:
        df: Dataframe level ketiga seri.
        maxlags: Batas atas pencarian lag.

    Returns:
        Lag VAR terpilih menurut AIC (minimum 1).
    """
    with warnings.catch_warnings():
        # Index tanggal asli tidak berfrekuensi tetap (kalender bursa); statsmodels
        # hanya memperingatkan bahwa frekuensi diabaikan, tidak memengaruhi estimasi.
        warnings.simplefilter("ignore", ValueWarning)
        selected = VAR(df).select_order(maxlags=maxlags)
    return max(int(selected.selected_orders["aic"]), 1)


def test_cointegration(
    df: pd.DataFrame,
    alpha: float = 0.05,
    johansen_det_order: int = 0,
    aux_maxlags: int = 10,
) -> dict[str, Any]:
    """Menguji kointegrasi pada level: Engle-Granger per pasangan dan Johansen.

    Dijalankan pada **level**, bukan log-return -- kointegrasi adalah relasi
    keseimbangan jangka panjang antar-level harga (mis. rasio emas-perak),
    yang justru hilang bila datanya dibedakan.

    Args:
        df: Dataframe level (kolom = seri, index = tanggal).
        alpha: Taraf nyata uji.
        johansen_det_order: Komponen deterministik Johansen
            (0 = konstanta di luar ruang kointegrasi).
        aux_maxlags: Batas atas pencarian lag bantu untuk menurunkan
            ``k_ar_diff`` Johansen (lihat :func:`_select_aux_lag`).

    Returns:
        Dictionary berisi hasil Engle-Granger tiap pasangan (kedua arah) dan
        hasil Johansen (trace statistic dan max-eigenvalue statistic) pada
        sistem seluruh kolom ``df``.
    """
    columns = list(df.columns)

    engle_granger: dict[str, Any] = {}
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            for y0_name, y1_name in ((col_a, col_b), (col_b, col_a)):
                stat, pvalue, crit_values = coint(
                    df[y0_name].to_numpy(), df[y1_name].to_numpy(), trend="c", autolag="aic"
                )
                key = f"{y0_name}~{y1_name}"
                engle_granger[key] = {
                    "y0": y0_name,
                    "y1": y1_name,
                    "statistic": float(stat),
                    "p_value": float(pvalue),
                    "critical_values": {
                        "1%": float(crit_values[0]),
                        "5%": float(crit_values[1]),
                        "10%": float(crit_values[2]),
                    },
                    "cointegrated": bool(pvalue < alpha),
                }

    aux_lag = _select_aux_lag(df, aux_maxlags)
    k_ar_diff = max(aux_lag - 1, 0)
    johansen = coint_johansen(df.to_numpy(), johansen_det_order, k_ar_diff)

    n_series = len(columns)
    # Kolom kritis statsmodels: [90%, 95%, 99%]
    crit_col = {0.10: 0, 0.05: 1, 0.01: 2}[min((0.10, 0.05, 0.01), key=lambda a: abs(a - alpha))]

    def _rank_table(stat: np.ndarray, crit: np.ndarray) -> tuple[list[dict[str, Any]], int]:
        table = []
        rank = 0
        rejected_all_lower = True
        for r in range(n_series):
            reject = bool(stat[r] > crit[r, crit_col])
            if reject and rejected_all_lower:
                rank = r + 1
            else:
                rejected_all_lower = False
            table.append(
                {
                    "r_less_equal": r,
                    "statistic": float(stat[r]),
                    "crit_90": float(crit[r, 0]),
                    "crit_95": float(crit[r, 1]),
                    "crit_99": float(crit[r, 2]),
                    "reject_h0": reject,
                }
            )
        return table, rank

    trace_table, rank_trace = _rank_table(johansen.trace_stat, johansen.trace_stat_crit_vals)
    max_eig_table, rank_max_eig = _rank_table(
        johansen.max_eig_stat, johansen.max_eig_stat_crit_vals
    )

    return {
        "alpha": alpha,
        "engle_granger": engle_granger,
        "johansen": {
            "series": columns,
            "det_order": johansen_det_order,
            "aux_lag_var": aux_lag,
            "k_ar_diff": k_ar_diff,
            "eigenvalues": [float(v) for v in johansen.eig],
            "trace_test": trace_table,
            "max_eig_test": max_eig_table,
            "rank_trace": rank_trace,
            "rank_max_eig": rank_max_eig,
        },
    }


# =============================================================================
# 5. Pemilihan lag VAR
# =============================================================================


def select_var_lag(df: pd.DataFrame, maxlags: int = 20) -> dict[str, Any]:
    """Memilih orde lag VAR (pada level) via AIC, BIC, HQIC, dan FPE.

    BIC dipakai untuk hasil utama karena lebih konservatif (menghindari
    overfitting lag pada seri yang relatif pendek); AIC dilaporkan sebagai
    pemeriksaan ketahanan. Bila keduanya jauh berbeda, kesenjangan itu
    dilaporkan apa adanya, bukan diselesaikan dengan memilih salah satu secara
    sepihak.

    Args:
        df: Dataframe level ketiga seri.
        maxlags: Batas atas pencarian lag.

    Returns:
        Dictionary berisi lag terpilih tiap kriteria, tabel lengkap nilai
        kriteria per lag, lag utama (BIC), dan lag pemeriksaan ketahanan (AIC).
    """
    with warnings.catch_warnings():
        # Index tanggal asli tidak berfrekuensi tetap (kalender bursa); statsmodels
        # hanya memperingatkan bahwa frekuensi diabaikan, tidak memengaruhi estimasi.
        warnings.simplefilter("ignore", ValueWarning)
        selected = VAR(df).select_order(maxlags=maxlags)

    criteria_table = [
        {
            "lag": lag,
            "aic": float(selected.ics["aic"][lag]),
            "bic": float(selected.ics["bic"][lag]),
            "hqic": float(selected.ics["hqic"][lag]),
            "fpe": float(selected.ics["fpe"][lag]),
        }
        for lag in range(len(selected.ics["aic"]))
    ]

    selected_orders = {
        criterion: int(value) for criterion, value in selected.selected_orders.items()
    }

    p_aic = selected_orders["aic"]
    p_bic = selected_orders["bic"]

    return {
        "maxlags": maxlags,
        "selected_orders": selected_orders,
        "criteria_table": criteria_table,
        "p_main": p_bic,
        "p_main_criterion": "bic",
        "p_robustness": p_aic,
        "p_robustness_criterion": "aic",
        "large_discrepancy": bool(abs(p_bic - p_aic) >= 3),
    }


# =============================================================================
# 6. Granger causality -- prosedur Toda-Yamamoto
# =============================================================================


def _build_lagged_design(df: pd.DataFrame, total_lags: int) -> pd.DataFrame:
    """Membangun matriks rancangan berisi lag 1..``total_lags`` tiap kolom ``df``."""
    lagged = {
        f"{column}__L{lag}": df[column].shift(lag)
        for column in df.columns
        for lag in range(1, total_lags + 1)
    }
    return pd.DataFrame(lagged, index=df.index)


def granger_toda_yamamoto(
    df: pd.DataFrame,
    p: int,
    d_max: int,
    robust: str = "HAC",
) -> dict[str, Any]:
    """Menguji presedensi prediktif dua arah dengan prosedur Toda-Yamamoto.

    Prosedur (Toda & Yamamoto, 1995), diimplementasikan eksplisit -- bukan
    memakai uji Granger standar bawaan statsmodels -- karena hasilnya harus
    tetap valid baik seri berkointegrasi maupun tidak:

    1. VAR diestimasi pada **level** dengan ``p + d_max`` lag (bukan ``p``).
    2. Untuk tiap pasangan (penyebab-uji, target), persamaan target diregresi
       terhadap lag 1..``p+d_max`` seluruh variabel dalam sistem (termasuk
       variabel ketiga sebagai kontrol).
    3. Uji Wald hanya membatasi ``p`` koefisien **pertama** (lag 1..``p``) dari
       variabel penyebab-uji -- lag tambahan ``d_max`` disertakan di model
       hanya sebagai penyesuai derajat integrasi, bukan bagian yang diuji.
    4. Uji Wald memakai kovarians HAC (Newey-West) karena log-return finansial
       bersifat heteroskedastik; tanpa koreksi ini p-value akan terlalu kecil.
       Lag HAC dipakai sebesar ``p + d_max`` (jumlah total lag pada model),
       konsisten dengan derajat ketergantungan temporal yang dimodelkan.
    5. Statistik Wald mengikuti distribusi chi-square dengan derajat bebas ``p``.

    Diuji dua arah untuk setiap pasangan kolom pada ``df`` (mis. untuk 3 kolom
    akan dihasilkan 6 pasangan terarah).

    Args:
        df: Dataframe level (kolom = seri, index = tanggal). Kolom yang tidak
            sedang diuji sebagai penyebab/target ikut dimasukkan sebagai
            kontrol (sistem VAR penuh).
        p: Jumlah lag yang diuji (dari pemilihan lag VAR pada data level).
        d_max: Ordo integrasi maksimum antar-seri (dari uji stasioneritas).
        robust: Jenis kovarians robust pada uji Wald. Hanya ``"HAC"`` yang
            didukung saat ini.

    Returns:
        Dictionary ``{"total_lags", "hac_maxlags", "pairs": {"x->y": {...}}}``
        berisi statistik Wald, derajat bebas, p-value, dan kesimpulan pada
        alfa 0.05 tiap pasangan terarah.

    Raises:
        RelationshipAnalysisError: Bila ``robust`` bukan ``"HAC"``, atau ``p``
            / ``d_max`` tidak valid.
    """
    if robust != "HAC":
        raise RelationshipAnalysisError(f"robust='{robust}' tidak didukung, hanya 'HAC'.")
    if p < 1:
        raise RelationshipAnalysisError(f"p harus >= 1, diterima {p}.")
    if d_max < 0:
        raise RelationshipAnalysisError(f"d_max harus >= 0, diterima {d_max}.")

    total_lags = p + d_max
    columns = list(df.columns)

    design = _build_lagged_design(df, total_lags)
    combined = pd.concat([df, design], axis=1).dropna()

    alpha = 0.05
    pairs: dict[str, Any] = {}

    for cause in columns:
        for effect in columns:
            if cause == effect:
                continue

            exog_columns = [
                f"{column}__L{lag}" for column in columns for lag in range(1, total_lags + 1)
            ]
            exog = add_constant(combined[exog_columns])
            endog = combined[effect]

            model = OLS(endog, exog)
            result = model.fit(cov_type="HAC", cov_kwds={"maxlags": total_lags})

            restriction = np.zeros((p, len(exog.columns)))
            for row, lag in enumerate(range(1, p + 1)):
                col_index = exog.columns.get_loc(f"{cause}__L{lag}")
                restriction[row, col_index] = 1.0

            wald = result.wald_test(restriction, use_f=False, scalar=True)
            statistic = float(np.asarray(wald.statistic).item())
            p_value = float(np.asarray(wald.pvalue).item())
            significant = p_value < alpha

            interpretation = (
                f"{_short_label(cause)} Granger-cause {_short_label(effect)} pada lag 1-{p} "
                f"(statistik Wald={statistic:.3f}, df={p}, p={p_value:.4f}): "
                + (
                    "presedensi prediktif signifikan pada alfa 0.05."
                    if significant
                    else "tidak ditemukan presedensi prediktif signifikan pada alfa 0.05."
                )
            )
            assert_no_causal_language(interpretation)

            pairs[f"{cause}->{effect}"] = {
                "cause": cause,
                "effect": effect,
                "p": p,
                "d_max": d_max,
                "total_lags": total_lags,
                "statistic": statistic,
                "df": p,
                "p_value": p_value,
                "alpha": alpha,
                "significant": significant,
                "interpretation": interpretation,
            }

    return {
        "total_lags": total_lags,
        "hac_maxlags": total_lags,
        "robust": robust,
        "pairs": pairs,
    }


# =============================================================================
# 7. Diagnostik residual VAR
# =============================================================================


def diagnose_var_residuals(
    var_result: VARResultsWrapper,
    alpha: float = 0.05,
    ljungbox_lags: int = 10,
    arch_lags: int = 10,
) -> dict[str, Any]:
    """Mendiagnosis residual model VAR: Ljung-Box dan ARCH-LM per persamaan.

    Ljung-Box menguji autokorelasi sisa (H0: residual tidak berautokorelasi).
    ARCH-LM menguji heteroskedastisitas bersyarat (H0: tidak ada efek ARCH).
    Bila Ljung-Box menolak H0 pada suatu variabel, model VAR pada variabel itu
    belum menangkap seluruh struktur temporalnya -- lag perlu ditambah dan
    model diestimasi ulang; peringatan dicetak ke log untuk kasus ini.

    Args:
        var_result: Hasil ``VAR(df).fit(p)`` dari statsmodels.
        alpha: Taraf nyata uji.
        ljungbox_lags: Jumlah lag yang diuji pada Ljung-Box.
        arch_lags: Jumlah lag yang diuji pada ARCH-LM.

    Returns:
        Dictionary berisi hasil Ljung-Box dan ARCH-LM tiap variabel, serta
        ringkasan apakah diagnostik keseluruhan lolos dan peringatan (bila ada).
    """
    residuals = var_result.resid
    warnings_list: list[str] = []
    per_variable: dict[str, Any] = {}

    for column in residuals.columns:
        resid_series = residuals[column].to_numpy()

        lb = acorr_ljungbox(resid_series, lags=[ljungbox_lags], return_df=True)
        lb_pvalue = float(lb["lb_pvalue"].iloc[0])
        lb_pass = lb_pvalue > alpha

        arch_stat, arch_pvalue, _, _ = het_arch(resid_series, nlags=arch_lags)
        arch_pass = float(arch_pvalue) > alpha

        if not lb_pass:
            warnings_list.append(
                f"Ljung-Box menolak H0 pada residual '{column}' (p={lb_pvalue:.4f}) -- "
                "otokorelasi sisa terdeteksi, pertimbangkan menambah lag VAR dan "
                "mengestimasi ulang model."
            )

        per_variable[column] = {
            "ljung_box": {
                "lags": ljungbox_lags,
                "statistic": float(lb["lb_stat"].iloc[0]),
                "p_value": lb_pvalue,
                "pass": lb_pass,
            },
            "arch_lm": {
                "lags": arch_lags,
                "statistic": float(arch_stat),
                "p_value": float(arch_pvalue),
                "pass": arch_pass,
            },
        }

    all_pass = all(
        entry["ljung_box"]["pass"] and entry["arch_lm"]["pass"] for entry in per_variable.values()
    )

    for message in warnings_list:
        print(f"PERINGATAN: {message}")

    return {
        "k_ar": int(var_result.k_ar),
        "alpha": alpha,
        "per_variable": per_variable,
        "all_pass": all_pass,
        "warnings": warnings_list,
    }


def fit_var_with_ljungbox_escalation(
    df: pd.DataFrame,
    p_start: int,
    maxlags: int,
    alpha: float,
    ljungbox_lags: int,
    arch_lags: int,
    logger: logging.Logger,
) -> tuple[int, VARResultsWrapper, dict[str, Any]]:
    """Mengestimasi VAR pada level, menambah lag bila Ljung-Box menolak H0.

    Model VAR yang dipakai untuk menguji presedensi prediktif (Toda-Yamamoto)
    hanya valid bila residualnya bebas dari autokorelasi sisa. Bila Ljung-Box
    menolak H0 pada lag awal (dipilih via AIC/BIC), lag dinaikkan satu per satu
    dan model diestimasi ulang sampai Ljung-Box tidak lagi menolak atau batas
    ``maxlags`` tercapai.

    Args:
        df: Dataframe level ketiga seri.
        p_start: Lag awal (dari :func:`select_var_lag`).
        maxlags: Batas atas kenaikan lag.
        alpha: Taraf nyata uji.
        ljungbox_lags: Jumlah lag yang diuji pada Ljung-Box.
        arch_lags: Jumlah lag yang diuji pada ARCH-LM.
        logger: Logger yang dipakai untuk mencatat tiap kenaikan lag.

    Returns:
        Tuple ``(p_final, var_result, diagnostics)`` -- lag akhir yang dipakai,
        hasil fit VAR pada lag itu, dan diagnostik residualnya. Bila batas
        ``maxlags`` tercapai tanpa Ljung-Box lolos, dikembalikan apa adanya
        pada lag terakhir yang dicoba (dilaporkan sebagai belum lolos).
    """
    p = p_start
    while True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ValueWarning)
            var_result = VAR(df).fit(p, trend="c")

        diagnostics = diagnose_var_residuals(
            var_result, alpha=alpha, ljungbox_lags=ljungbox_lags, arch_lags=arch_lags
        )
        ljungbox_pass = all(
            entry["ljung_box"]["pass"] for entry in diagnostics["per_variable"].values()
        )

        if ljungbox_pass or p >= maxlags:
            return p, var_result, diagnostics

        logger.warning(
            "Ljung-Box menolak H0 pada lag %d -- menambah lag menjadi %d dan "
            "mengestimasi ulang VAR.",
            p,
            p + 1,
        )
        p += 1


# =============================================================================
# Orkestrasi
# =============================================================================


def _relative_to_project(path: Path) -> str:
    """Mengubah path absolut menjadi path relatif terhadap akar project."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_train_data(
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Memuat data latih dari ``data/processed/train.csv``.

    Hanya berkas latih yang dibaca -- analisis hubungan adalah tujuan
    penelitian mandiri yang hasilnya tidak boleh dibocori oleh data validasi
    maupun data uji.

    Args:
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dataframe data latih dengan index tanggal, hanya kolom harga
        (kolom flag ``*_ffilled`` dibuang).

    Raises:
        FileNotFoundError: Bila berkas latih belum ada.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("relationship_analysis")

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

    target_column = get_target_column(config)
    covariate_columns = get_covariate_columns(config)
    frame = frame.loc[:, [target_column, *covariate_columns]]

    logger.info(
        "Memuat data latih: %d baris (%s s.d. %s) dari %s",
        len(frame),
        frame.index[0].date(),
        frame.index[-1].date(),
        path,
    )
    logger.info("Data validasi dan data uji TIDAK dibaca (mencegah kebocoran data).")

    return frame


def _load_mutual_information(config: dict[str, Any], logger: logging.Logger) -> dict[str, Any] | None:
    """Memuat hasil Mutual Information sebagai pelengkap non-linear.

    Args:
        config: Konfigurasi project.
        logger: Logger yang dipakai.

    Returns:
        Isi ``results/metrics/mutual_information.json``, atau ``None`` bila
        berkas belum ada (analisis Granger tetap berjalan tanpa pembanding MI).
    """
    path = resolve_path(config["paths"]["metrics_dir"]) / "mutual_information.json"
    if not path.is_file():
        logger.warning(
            "Berkas Mutual Information '%s' tidak ditemukan -- pembanding non-linear "
            "dilewati. Jalankan 'python -m src.mutual_information' agar tersedia.",
            path,
        )
        return None

    import json

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_markdown_summary(
    stationarity: dict[str, Any],
    cointegration: dict[str, Any],
    lag_selection: dict[str, Any],
    granger: dict[str, Any],
    diagnostics: dict[str, Any],
    mi_report: dict[str, Any] | None,
    target_column: str,
    covariate_columns: list[str],
) -> str:
    """Menyusun tabel ringkas format markdown, siap disalin ke naskah skripsi.

    Args:
        stationarity: ``{nama_seri: hasil test_stationarity}``.
        cointegration: Hasil :func:`test_cointegration`.
        lag_selection: Hasil :func:`select_var_lag`.
        granger: Hasil :func:`granger_toda_yamamoto`.
        diagnostics: Hasil :func:`diagnose_var_residuals`.
        mi_report: Isi ``mutual_information.json``, atau ``None``.
        target_column: Nama kolom target (perak).
        covariate_columns: Nama kolom kovariat (emas, dxy).

    Returns:
        String markdown multi-tabel.
    """
    lines: list[str] = []

    lines.append("### Tabel 1. Ordo integrasi tiap seri (data latih)")
    lines.append("")
    lines.append("| Seri | Verdict level | Verdict log-return | Ordo integrasi (d) | Catatan |")
    lines.append("|---|---|---|---:|---|")
    for name, result in stationarity.items():
        d = result["order_of_integration"]["d"]
        d_text = "-" if d is None else str(d)
        lines.append(
            f"| {_short_label(name)} | {result['level']['verdict']} | "
            f"{result['log_return']['verdict']} | {d_text} | "
            f"{result['order_of_integration']['note']} |"
        )

    lines.append("")
    lines.append("### Tabel 2. Kointegrasi Engle-Granger dan Johansen (level, data latih)")
    lines.append("")
    lines.append("| Pasangan (y0~y1) | Statistik EG | p-value EG | Kointegrasi (alfa=0.05) |")
    lines.append("|---|---:|---:|---|")
    for key, result in cointegration["engle_granger"].items():
        lines.append(
            f"| {key} | {result['statistic']:.4f} | {result['p_value']:.4f} | "
            f"{'ya' if result['cointegrated'] else 'tidak'} |"
        )
    johansen = cointegration["johansen"]
    lines.append("")
    lines.append(
        f"Johansen (sistem {', '.join(_short_label(c) for c in johansen['series'])}, "
        f"k_ar_diff={johansen['k_ar_diff']}): rank kointegrasi trace test = "
        f"{johansen['rank_trace']}, rank max-eigenvalue test = {johansen['rank_max_eig']}."
    )

    lines.append("")
    lines.append("### Tabel 3. Pemilihan lag VAR (level, data latih)")
    lines.append("")
    lines.append(
        f"Lag utama (BIC) = {lag_selection['p_main']}; lag pemeriksaan ketahanan (AIC) = "
        f"{lag_selection['p_robustness']}."
        + (
            " Kesenjangan antar-kriteria besar (>= 3 lag)."
            if lag_selection["large_discrepancy"]
            else ""
        )
        + (
            f" Lag dinaikkan menjadi {diagnostics['p_final']} pada tahap diagnostik "
            "residual karena Ljung-Box menolak H0 pada lag BIC; lag akhir inilah yang "
            "dipakai pada uji presedensi prediktif (Tabel 4)."
            if diagnostics["escalated"]
            else " Lag ini lolos diagnostik residual (Tabel 5) dan dipakai langsung "
            "pada uji presedensi prediktif (Tabel 4)."
        )
    )
    lines.append("")
    lines.append("| Lag | AIC | BIC | HQIC | FPE |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in lag_selection["criteria_table"]:
        lines.append(
            f"| {row['lag']} | {row['aic']:.4f} | {row['bic']:.4f} | "
            f"{row['hqic']:.4f} | {row['fpe']:.4f} |"
        )

    lines.append("")
    lines.append(
        "### Tabel 4. Presedensi prediktif Toda-Yamamoto vs Mutual Information berlag"
    )
    lines.append("")
    lines.append(
        f"Diuji pada lag akhir p = {diagnostics['p_final']} (lihat catatan Tabel 3), "
        f"dengan total lag VAR p + d_max = {granger['total_lags']} pada langkah "
        "Toda-Yamamoto."
    )
    lines.append("")
    lines.append(
        "| Pasangan (X -> Y) | Lag diuji | Statistik Wald | df | p-value | Kesimpulan (alfa=0.05) | MI (log-return, lag>=1) |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---:|")

    mi_lag_lookup: dict[str, float] = {}
    if mi_report is not None:
        lagged_log_return = mi_report.get("lagged", {}).get("log_return", {}).get("pairs", {})
        for covariate, series in lagged_log_return.items():
            positive_lags = [point["mi_mean"] for point in series if point["lag"] > 0]
            if positive_lags:
                mi_lag_lookup[covariate] = max(positive_lags)

    relevant_pairs = [
        f"{covariate}->{target_column}" for covariate in covariate_columns
    ] + [f"{target_column}->{covariate}" for covariate in covariate_columns]

    for pair_key in relevant_pairs:
        result = granger["pairs"].get(pair_key)
        if result is None:
            continue
        mi_value = mi_lag_lookup.get(result["cause"])
        mi_text = f"{mi_value:.4f}" if mi_value is not None else "-"
        lines.append(
            f"| {_short_label(result['cause'])} -> {_short_label(result['effect'])} | "
            f"1-{result['p']} | {result['statistic']:.4f} | {result['df']} | "
            f"{result['p_value']:.4f} | "
            f"{'Granger-cause' if result['significant'] else 'tidak signifikan'} | "
            f"{mi_text} |"
        )

    lines.append("")
    lines.append(
        "*Kolom MI diambil dari MI maksimum pada lag >= 1 varian log-return "
        "(`results/metrics/mutual_information.json`), sebagai pembanding non-linear "
        "terhadap hasil Granger yang bersifat linear. Ketidaksesuaian antara "
        "keduanya (mis. MI tinggi tapi Granger tidak signifikan) adalah temuan "
        "yang menunjukkan keterkaitan bersifat non-linear atau sezaman, bukan "
        "berlag secara linear.*"
    )

    lines.append("")
    lines.append(f"### Tabel 5. Diagnostik residual VAR (lag akhir p = {diagnostics['p_final']})")
    lines.append("")
    first_entry = next(iter(diagnostics["per_variable"].values()))
    lb_lag = first_entry["ljung_box"]["lags"]
    arch_lag = first_entry["arch_lm"]["lags"]
    lines.append(
        f"| Variabel | Ljung-Box p (lag={lb_lag}) | Lolos | ARCH-LM p (lag={arch_lag}) | Lolos |"
    )
    lines.append("|---|---:|---|---:|---|")
    for column, entry in diagnostics["per_variable"].items():
        lines.append(
            f"| {_short_label(column)} | {entry['ljung_box']['p_value']:.4f} | "
            f"{'ya' if entry['ljung_box']['pass'] else 'tidak'} | "
            f"{entry['arch_lm']['p_value']:.4f} | "
            f"{'ya' if entry['arch_lm']['pass'] else 'tidak'} |"
        )
    ljungbox_all_pass = all(
        entry["ljung_box"]["pass"] for entry in diagnostics["per_variable"].values()
    )
    arch_all_pass = all(
        entry["arch_lm"]["pass"] for entry in diagnostics["per_variable"].values()
    )
    lines.append("")
    lines.append(
        f"Diagnostik keseluruhan: {'lolos' if diagnostics['all_pass'] else 'TIDAK lolos'}."
    )
    if diagnostics["escalated"]:
        lines.append(
            f"Lag dinaikkan dari {diagnostics['p_main']} menjadi {diagnostics['p_final']} "
            "selama eskalasi Ljung-Box"
            + (
                f", dan berhasil lolos pada lag {diagnostics['p_final']}."
                if ljungbox_all_pass
                else f", tetapi Ljung-Box masih menolak H0 pada batas lag "
                f"yang diizinkan ({diagnostics['p_final']})."
            )
        )
    if ljungbox_all_pass and not arch_all_pass:
        lines.append(
            "ARCH-LM tetap menolak H0 (heteroskedastisitas bersyarat) pada seluruh "
            "variabel meski Ljung-Box sudah lolos -- wajar untuk data finansial dan "
            "bukan indikasi kesalahan spesifikasi lag; menambah lag VAR tidak "
            "menghilangkan efek ARCH. Uji Wald pada langkah presedensi prediktif "
            "sudah memakai kovarians HAC untuk mengantisipasi hal ini."
        )
    if diagnostics["warnings"]:
        lines.append("")
        for message in diagnostics["warnings"]:
            lines.append(f"> PERINGATAN: {message}")

    return "\n".join(lines)


def run_relationship_analysis(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Menjalankan seluruh analisis hubungan antar-variabel secara berurutan.

    Urutan (tiap langkah menentukan validitas langkah berikutnya):
    stasioneritas -> ACF/PACF -> cross-correlation -> kointegrasi ->
    pemilihan lag VAR -> diagnostik residual (dengan eskalasi lag bila
    Ljung-Box menolak H0) -> Granger causality (Toda-Yamamoto) pada lag hasil
    diagnostik -> Mutual Information sebagai pelengkap non-linear.

    Args:
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.

    Returns:
        Dictionary laporan lengkap yang juga disimpan ke JSON.
    """
    if config is None:
        config = load_config()

    logger = setup_logger(
        "relationship_analysis", log_dir=config.get("paths", {}).get("logs_dir", "logs")
    )
    settings = config["relationship_analysis"]
    alpha = settings["alpha"]
    figures_dir = config["paths"]["figures_dir"]

    target_column = get_target_column(config)
    covariate_columns = get_covariate_columns(config)

    logger.info("=" * 78)
    logger.info("ANALISIS HUBUNGAN ANTAR-VARIABEL -- hanya memakai data latih")
    logger.info("=" * 78)

    train = load_train_data(config=config, logger=logger)
    all_columns = [target_column, *covariate_columns]

    # --- 1. Stasioneritas -----------------------------------------------------
    logger.info("-" * 78)
    logger.info("Langkah 1/7: Uji stasioneritas (ADF & KPSS)")
    stationarity: dict[str, Any] = {}
    for column in all_columns:
        result = test_stationarity(
            train[column],
            column,
            alpha=alpha,
            regression=settings["stationarity_regression"],
        )
        stationarity[column] = result
        logger.info(
            "  %-14s level=%-14s log_return=%-14s d=%s",
            column,
            result["level"]["verdict"],
            result["log_return"]["verdict"],
            result["order_of_integration"]["d"],
        )

    d_values = [r["order_of_integration"]["d"] for r in stationarity.values()]
    unresolved = [name for name, r in stationarity.items() if r["order_of_integration"]["d"] is None]
    if unresolved:
        logger.warning(
            "Ordo integrasi tidak konklusif untuk %s -- dipakai d=1 sebagai asumsi "
            "kerja konservatif (konvensi umum harga finansial adalah I(1)).",
            unresolved,
        )
    d_max = max((d for d in d_values if d is not None), default=1)
    d_max = max(d_max, 1 if unresolved else 0)
    logger.info("d_max (ordo integrasi maksimum) = %d", d_max)

    # --- 2. ACF/PACF ------------------------------------------------------------
    logger.info("-" * 78)
    logger.info("Langkah 2/7: ACF dan PACF (log-return)")
    acf_pacf_path = plot_acf_pacf(
        train,
        mode=settings["acf_pacf"]["mode"],
        lags=settings["acf_pacf"]["lags"],
        output_path=Path(figures_dir) / "relationship" / "acf_pacf.png",
        dpi=settings["figure_dpi"],
    )
    logger.info("Gambar tersimpan: %s", acf_pacf_path)

    # --- 3. Cross-correlation ----------------------------------------------------
    logger.info("-" * 78)
    logger.info("Langkah 3/7: Cross-correlation function (perak vs tiap kovariat, log-return)")
    # Dihitung pada log-return, bukan level -- harga level non-stasioner (langkah 1)
    # berbagi tren bersama, sehingga CCF level akan tinggi semu di seluruh lag dan
    # tidak informatif (persis alasan ACF/PACF juga dihitung pada log-return).
    log_returns = pd.concat({column: _log_return(train[column]) for column in all_columns}, axis=1)

    ccf_results: dict[str, Any] = {}
    ccf_figure_names = {covariate_columns[0]: "ccf_perak_emas.png"}
    if len(covariate_columns) > 1:
        ccf_figure_names[covariate_columns[1]] = "ccf_perak_dxy.png"

    for covariate in covariate_columns:
        ccf_df = cross_correlation(
            log_returns[covariate],
            log_returns[target_column],
            max_lag=settings["cross_correlation"]["max_lag"],
        )
        ccf_results[covariate] = ccf_df.to_dict(orient="records")

        figure_name = ccf_figure_names.get(covariate, f"ccf_perak_{covariate}.png")
        ccf_path = plot_cross_correlation(
            ccf_df,
            x_label=_short_label(covariate),
            y_label=_short_label(target_column),
            output_path=Path(figures_dir) / "relationship" / figure_name,
            dpi=settings["figure_dpi"],
        )
        logger.info("  CCF %s vs %s -> %s", covariate, target_column, ccf_path)

    # --- 4. Kointegrasi -----------------------------------------------------------
    logger.info("-" * 78)
    logger.info("Langkah 4/7: Kointegrasi (Engle-Granger & Johansen, level)")
    cointegration = test_cointegration(
        train,
        alpha=alpha,
        johansen_det_order=settings["cointegration"]["johansen_det_order"],
        aux_maxlags=settings["cointegration"]["aux_maxlags"],
    )
    for key, result in cointegration["engle_granger"].items():
        logger.info(
            "  Engle-Granger %-30s stat=%.4f p=%.4f kointegrasi=%s",
            key,
            result["statistic"],
            result["p_value"],
            result["cointegrated"],
        )
    logger.info(
        "  Johansen: rank trace=%d, rank max-eig=%d (k_ar_diff=%d)",
        cointegration["johansen"]["rank_trace"],
        cointegration["johansen"]["rank_max_eig"],
        cointegration["johansen"]["k_ar_diff"],
    )

    # --- 5. Pemilihan lag VAR -------------------------------------------------------
    logger.info("-" * 78)
    logger.info("Langkah 5/7: Pemilihan lag VAR (AIC/BIC/HQIC/FPE)")
    lag_selection = select_var_lag(train, maxlags=settings["var_lag_selection"]["maxlags"])
    logger.info(
        "  p_main (BIC)=%d, p_robustness (AIC)=%d, kesenjangan_besar=%s",
        lag_selection["p_main"],
        lag_selection["p_robustness"],
        lag_selection["large_discrepancy"],
    )

    # --- 6. Diagnostik residual VAR (dengan eskalasi lag bila Ljung-Box menolak) -----
    logger.info("-" * 78)
    logger.info("Langkah 6/7: Diagnostik residual VAR (Ljung-Box & ARCH-LM)")
    p_main = lag_selection["p_main"]
    p_final, var_fit, diagnostics = fit_var_with_ljungbox_escalation(
        train,
        p_start=p_main,
        maxlags=settings["var_lag_selection"]["maxlags"],
        alpha=alpha,
        ljungbox_lags=settings["diagnostics"]["ljungbox_lags"],
        arch_lags=settings["diagnostics"]["arch_lags"],
        logger=logger,
    )
    diagnostics["p_main"] = p_main
    diagnostics["p_final"] = p_final
    diagnostics["escalated"] = p_final != p_main
    if diagnostics["escalated"]:
        logger.info(
            "  Lag VAR dinaikkan dari %d (BIC) menjadi %d karena diagnostik Ljung-Box "
            "awal menolak H0.",
            p_main,
            p_final,
        )
    logger.info("  Diagnostik keseluruhan lolos: %s (lag akhir=%d)", diagnostics["all_pass"], p_final)

    # --- 7. Granger causality (Toda-Yamamoto) ----------------------------------------
    # Dipakai lag akhir (p_final) dari langkah 6, bukan p_main mentah, karena
    # spesifikasi VAR yang residualnya masih berautokorelasi tidak valid dipakai
    # untuk menguji presedensi prediktif.
    logger.info("-" * 78)
    logger.info("Langkah 7/7: Granger causality -- prosedur Toda-Yamamoto")
    granger = granger_toda_yamamoto(train, p=p_final, d_max=d_max, robust="HAC")
    for pair_key, result in granger["pairs"].items():
        logger.info(
            "  %-30s stat=%.4f df=%d p=%.4f signifikan=%s",
            pair_key,
            result["statistic"],
            result["df"],
            result["p_value"],
            result["significant"],
        )

    # --- Mutual Information sebagai pelengkap non-linear -----------------------------
    mi_report = _load_mutual_information(config, logger)

    markdown_summary = build_markdown_summary(
        stationarity,
        cointegration,
        lag_selection,
        granger,
        diagnostics,
        mi_report,
        target_column,
        covariate_columns,
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "Seluruh analisis hubungan dihitung hanya dari data latih.",
        "data": {
            "source": "data/processed/train.csv",
            "n_rows": int(len(train)),
            "date_start": str(train.index[0].date()),
            "date_end": str(train.index[-1].date()),
            "target": target_column,
            "covariates": covariate_columns,
        },
        "d_max": d_max,
        "stationarity": stationarity,
        "acf_pacf_figure": _relative_to_project(acf_pacf_path),
        "cross_correlation": ccf_results,
        "cointegration": cointegration,
        "lag_selection": lag_selection,
        "granger_toda_yamamoto": granger,
        "var_residual_diagnostics": diagnostics,
        "mutual_information_available": mi_report is not None,
    }

    metrics_dir = resolve_path(config["paths"]["metrics_dir"])
    json_path = save_json(report, metrics_dir / "relationship_analysis.json")
    logger.info("Hasil lengkap tersimpan di %s", json_path)

    summary_path = metrics_dir / "relationship_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(markdown_summary, encoding="utf-8")
    logger.info("Ringkasan markdown tersimpan di %s", summary_path)

    print()
    print(markdown_summary)
    print()

    return report


def main() -> dict[str, Any]:
    """Titik masuk eksekusi mandiri modul."""
    return run_relationship_analysis()


if __name__ == "__main__":
    main()
