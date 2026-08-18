"""Metrik evaluasi peramalan probabilistik.

Modul ini berisi seluruh metrik yang dipakai untuk membandingkan Chronos-2
zero-shot, Chronos-2 fine-tuned, dan baseline naive persistence. Modul sengaja
dibangun **sebelum** modul model, agar kebenarannya dapat diuji lebih dulu
memakai baseline dan data sintetis yang jawabannya sudah diketahui.

Konvensi bentuk array (dipakai seluruh fungsi publik):

* ``y_true``            : ``(n_windows, H)``
* ``y_pred``            : ``(n_windows, H)``            — ramalan titik (median)
* ``y_pred_quantiles``  : ``(n_windows, H, n_quantiles)``
* ``quantile_levels``   : daftar ``n_quantiles`` level, menaik, dalam (0, 1)

``n_windows`` adalah banyaknya jendela *rolling origin* (keputusan D4) dan
``H`` panjang horizon. Karena jendela saling tumpang tindih, uji signifikansi
Diebold-Mariano wajib memakai koreksi HAC — lihat :func:`diebold_mariano`.

Metrik yang tersedia:

* Titik        : :func:`mae`, :func:`rmse`, :func:`mape`, :func:`mase`
* Probabilistik: :func:`quantile_loss`, :func:`crps_approx`,
  :func:`normalized_crps`, :func:`coverage`, :func:`mean_interval_width`,
  :func:`quantile_crossing_rate`
* Signifikansi : :func:`diebold_mariano`
* Agregasi     : :func:`aggregate_metrics`

Cara menjalankan mandiri (mengevaluasi baseline pada periode validasi):
    python -m src.evaluation
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats

from src.preprocessing import get_target_column
from src.utils import load_config, resolve_path, save_json, setup_logger

# Toleransi pencocokan level kuantil terhadap nilai yang diminta
QUANTILE_MATCH_TOLERANCE = 1e-9


class EvaluationError(ValueError):
    """Kegagalan pada perhitungan metrik evaluasi."""


# =============================================================================
# Validasi bentuk array
# =============================================================================


def _as_2d(array: Any, name: str) -> np.ndarray:
    """Memvalidasi sebuah array bernilai ``(n_windows, H)``.

    Args:
        array: Array yang akan divalidasi.
        name: Nama argumen, dipakai pada pesan galat.

    Returns:
        Array float 2-D.

    Raises:
        EvaluationError: Bila dimensi bukan 2 atau ada nilai tidak berhingga.
    """
    values = np.asarray(array, dtype=float)
    if values.ndim != 2:
        raise EvaluationError(
            f"{name} harus berbentuk (n_windows, H), diterima {values.shape}."
        )
    if not np.isfinite(values).all():
        raise EvaluationError(f"{name} memuat NaN atau nilai tak berhingga.")
    return values


def _as_3d(array: Any, name: str) -> np.ndarray:
    """Memvalidasi sebuah array bernilai ``(n_windows, H, n_quantiles)``.

    Args:
        array: Array yang akan divalidasi.
        name: Nama argumen, dipakai pada pesan galat.

    Returns:
        Array float 3-D.

    Raises:
        EvaluationError: Bila dimensi bukan 3 atau ada nilai tidak berhingga.
    """
    values = np.asarray(array, dtype=float)
    if values.ndim != 3:
        raise EvaluationError(
            f"{name} harus berbentuk (n_windows, H, n_quantiles), "
            f"diterima {values.shape}."
        )
    if not np.isfinite(values).all():
        raise EvaluationError(f"{name} memuat NaN atau nilai tak berhingga.")
    return values


def _check_same_shape(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    """Memastikan dua array berbentuk sama.

    Args:
        a: Array pertama.
        b: Array kedua.
        name_a: Nama argumen pertama.
        name_b: Nama argumen kedua.

    Raises:
        EvaluationError: Bila bentuknya berbeda.
    """
    if a.shape != b.shape:
        raise EvaluationError(
            f"Bentuk {name_a} {a.shape} tidak sama dengan {name_b} {b.shape}."
        )


def _validate_quantile_levels(quantile_levels: Any, n_quantiles: int) -> np.ndarray:
    """Memvalidasi daftar level kuantil terhadap dimensi terakhir ramalan.

    Args:
        quantile_levels: Daftar level kuantil.
        n_quantiles: Banyaknya kuantil pada array ramalan.

    Returns:
        Array level kuantil 1-D.

    Raises:
        EvaluationError: Bila panjang tidak cocok, ada level di luar (0, 1),
            atau daftarnya tidak menaik tegas.
    """
    levels = np.asarray(quantile_levels, dtype=float)

    if levels.ndim != 1 or levels.size != n_quantiles:
        raise EvaluationError(
            f"quantile_levels harus 1-D sepanjang {n_quantiles}, "
            f"diterima bentuk {levels.shape}."
        )
    if ((levels <= 0.0) | (levels >= 1.0)).any():
        raise EvaluationError(
            f"Seluruh level kuantil harus di (0, 1), diterima {list(levels)}."
        )
    if not np.all(np.diff(levels) > 0):
        raise EvaluationError(
            f"quantile_levels harus menaik tegas, diterima {list(levels)}."
        )

    return levels


def find_quantile_index(quantile_levels: Any, level: float) -> int:
    """Mencari indeks kolom sebuah level kuantil.

    Args:
        quantile_levels: Daftar level kuantil.
        level: Level yang dicari, mis. ``0.5``.

    Returns:
        Indeks kolom pada dimensi kuantil.

    Raises:
        EvaluationError: Bila level tidak tersedia pada daftar.
    """
    levels = np.asarray(quantile_levels, dtype=float)
    matches = np.flatnonzero(np.isclose(levels, level, atol=QUANTILE_MATCH_TOLERANCE))

    if matches.size == 0:
        raise EvaluationError(
            f"Level kuantil {level} tidak ada pada daftar {list(levels)}."
        )

    return int(matches[0])


def interval_indices(quantile_levels: Any, coverage_level: float) -> tuple[int, int]:
    """Mencari pasangan indeks kuantil pembentuk interval prediksi simetris.

    Interval ``coverage_level`` dibentuk dari kuantil ``(1 - c) / 2`` dan
    ``1 - (1 - c) / 2``. Contoh: interval 80% memakai kuantil 0.1 dan 0.9,
    interval 95% memakai 0.025 dan 0.975 (keputusan D5).

    Args:
        quantile_levels: Daftar level kuantil.
        coverage_level: Tingkat cakupan yang diinginkan, mis. ``0.80``.

    Returns:
        Tuple ``(indeks_bawah, indeks_atas)``.

    Raises:
        EvaluationError: Bila ``coverage_level`` di luar (0, 1) atau kuantil
            yang dibutuhkan tidak tersedia.
    """
    if not 0.0 < coverage_level < 1.0:
        raise EvaluationError(
            f"coverage_level harus di (0, 1), diterima {coverage_level}."
        )

    tail = (1.0 - coverage_level) / 2.0
    return (
        find_quantile_index(quantile_levels, tail),
        find_quantile_index(quantile_levels, 1.0 - tail),
    )


# =============================================================================
# Metrik akurasi titik
# =============================================================================


def absolute_errors(y_true: Any, y_pred: Any) -> np.ndarray:
    """Menghitung galat absolut per elemen.

    Args:
        y_true: Nilai aktual ``(n_windows, H)``.
        y_pred: Ramalan titik ``(n_windows, H)``.

    Returns:
        Array ``(n_windows, H)`` berisi ``|y_true - y_pred|``.
    """
    actual = _as_2d(y_true, "y_true")
    predicted = _as_2d(y_pred, "y_pred")
    _check_same_shape(actual, predicted, "y_true", "y_pred")

    return np.abs(actual - predicted)


def mae(y_true: Any, y_pred: Any) -> float:
    """Menghitung *Mean Absolute Error*.

    Args:
        y_true: Nilai aktual ``(n_windows, H)``.
        y_pred: Ramalan titik ``(n_windows, H)``.

    Returns:
        Rata-rata galat absolut atas seluruh jendela dan horizon.
    """
    return float(absolute_errors(y_true, y_pred).mean())


def rmse(y_true: Any, y_pred: Any) -> float:
    """Menghitung *Root Mean Squared Error*.

    Lebih sensitif terhadap galat besar dibanding MAE, sehingga berguna untuk
    mendeteksi kegagalan ramalan pada periode bergejolak.

    Args:
        y_true: Nilai aktual ``(n_windows, H)``.
        y_pred: Ramalan titik ``(n_windows, H)``.

    Returns:
        Akar rata-rata galat kuadrat.
    """
    actual = _as_2d(y_true, "y_true")
    predicted = _as_2d(y_pred, "y_pred")
    _check_same_shape(actual, predicted, "y_true", "y_pred")

    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def absolute_percentage_errors(y_true: Any, y_pred: Any) -> np.ndarray:
    """Menghitung galat persentase absolut per elemen (dalam persen).

    Args:
        y_true: Nilai aktual ``(n_windows, H)``.
        y_pred: Ramalan titik ``(n_windows, H)``.

    Returns:
        Array ``(n_windows, H)`` berisi ``100 * |y_true - y_pred| / |y_true|``.

    Raises:
        EvaluationError: Bila ada nilai aktual nol (MAPE tidak terdefinisi).
    """
    actual = _as_2d(y_true, "y_true")
    predicted = _as_2d(y_pred, "y_pred")
    _check_same_shape(actual, predicted, "y_true", "y_pred")

    if np.isclose(actual, 0.0).any():
        raise EvaluationError(
            "y_true memuat nilai nol sehingga MAPE tidak terdefinisi."
        )

    return 100.0 * np.abs((actual - predicted) / actual)


def mape(y_true: Any, y_pred: Any) -> float:
    """Menghitung *Mean Absolute Percentage Error* dalam persen.

    Args:
        y_true: Nilai aktual ``(n_windows, H)``.
        y_pred: Ramalan titik ``(n_windows, H)``.

    Returns:
        Rata-rata galat persentase absolut (satuan persen).
    """
    return float(absolute_percentage_errors(y_true, y_pred).mean())


def mase(y_true: Any, y_pred: Any, y_naive: Any) -> float:
    """Menghitung MASE relatif terhadap baseline naive persistence.

    Didefinisikan sebagai ``MAE(model) / MAE(naive)`` dengan naive dievaluasi
    pada **himpunan jendela yang persis sama** (keputusan D4). Tafsirannya
    langsung: nilai < 1 berarti model mengalahkan baseline, nilai > 1 berarti
    kalah, dan tepat 1 berarti setara.

    Penyebutnya sengaja memakai galat baseline pada jendela uji yang sama,
    bukan galat *in-sample* seperti definisi MASE musiman Hyndman, agar
    perbandingan zero-shot vs fine-tuned vs baseline berada pada dasar identik.

    Args:
        y_true: Nilai aktual ``(n_windows, H)``.
        y_pred: Ramalan titik model ``(n_windows, H)``.
        y_naive: Ramalan titik baseline ``(n_windows, H)``.

    Returns:
        Rasio MAE model terhadap MAE baseline.

    Raises:
        EvaluationError: Bila MAE baseline nol (rasio tidak terdefinisi).
    """
    numerator = mae(y_true, y_pred)
    denominator = mae(y_true, y_naive)

    if np.isclose(denominator, 0.0):
        raise EvaluationError(
            "MAE baseline bernilai nol sehingga MASE tidak terdefinisi."
        )

    return float(numerator / denominator)


# =============================================================================
# Metrik probabilistik
# =============================================================================


def quantile_loss_elementwise(y_true: Any, y_pred_q: Any, tau: float) -> np.ndarray:
    """Menghitung *pinball loss* per elemen untuk satu level kuantil.

    Rumus (sesuai CLAUDE.md)::

        QL_tau(y, yhat) = max(tau * (y - yhat), (tau - 1) * (y - yhat))

    Bentuk ini menghukum *under-prediction* dengan bobot ``tau`` dan
    *over-prediction* dengan bobot ``1 - tau``, sehingga kuantil tinggi lebih
    dihukum bila ramalannya terlalu rendah.

    Args:
        y_true: Nilai aktual ``(n_windows, H)``.
        y_pred_q: Ramalan pada satu level kuantil ``(n_windows, H)``.
        tau: Level kuantil, di dalam (0, 1).

    Returns:
        Array ``(n_windows, H)`` berisi pinball loss.

    Raises:
        EvaluationError: Bila ``tau`` di luar (0, 1).
    """
    if not 0.0 < tau < 1.0:
        raise EvaluationError(f"tau harus di (0, 1), diterima {tau}.")

    actual = _as_2d(y_true, "y_true")
    predicted = _as_2d(y_pred_q, "y_pred_q")
    _check_same_shape(actual, predicted, "y_true", "y_pred_q")

    difference = actual - predicted
    return np.maximum(tau * difference, (tau - 1.0) * difference)


def quantile_loss(y_true: Any, y_pred_q: Any, tau: float) -> float:
    """Menghitung rata-rata *pinball loss* untuk satu level kuantil.

    Args:
        y_true: Nilai aktual ``(n_windows, H)``.
        y_pred_q: Ramalan pada satu level kuantil ``(n_windows, H)``.
        tau: Level kuantil, di dalam (0, 1).

    Returns:
        Rata-rata pinball loss atas seluruh jendela dan horizon.
    """
    return float(quantile_loss_elementwise(y_true, y_pred_q, tau).mean())


def crps_elementwise(
    y_true: Any, y_pred_quantiles: Any, quantile_levels: Any
) -> np.ndarray:
    """Menghitung aproksimasi CRPS per elemen dari kumpulan kuantil.

    Rumus (sesuai CLAUDE.md)::

        CRPS ~= (2 / K) * sum_k QL_{tau_k}

    dengan K banyaknya level kuantil. Faktor 2 membuat besarannya sebanding
    dengan CRPS sesungguhnya; ketika ramalan runtuh menjadi satu titik, nilainya
    kembali menjadi galat absolut.

    Args:
        y_true: Nilai aktual ``(n_windows, H)``.
        y_pred_quantiles: Ramalan kuantil ``(n_windows, H, n_quantiles)``.
        quantile_levels: Daftar level kuantil, menaik tegas.

    Returns:
        Array ``(n_windows, H)`` berisi aproksimasi CRPS.

    Raises:
        EvaluationError: Bila bentuk array tidak konsisten.
    """
    actual = _as_2d(y_true, "y_true")
    predicted = _as_3d(y_pred_quantiles, "y_pred_quantiles")

    if actual.shape != predicted.shape[:2]:
        raise EvaluationError(
            f"Dua dimensi pertama y_pred_quantiles {predicted.shape[:2]} tidak "
            f"sama dengan bentuk y_true {actual.shape}."
        )

    levels = _validate_quantile_levels(quantile_levels, predicted.shape[2])

    difference = actual[:, :, None] - predicted
    losses = np.maximum(
        levels[None, None, :] * difference, (levels[None, None, :] - 1.0) * difference
    )

    return (2.0 / levels.size) * losses.sum(axis=2)


def crps_approx(y_true: Any, y_pred_quantiles: Any, quantile_levels: Any) -> float:
    """Menghitung rata-rata aproksimasi CRPS.

    Args:
        y_true: Nilai aktual ``(n_windows, H)``.
        y_pred_quantiles: Ramalan kuantil ``(n_windows, H, n_quantiles)``.
        quantile_levels: Daftar level kuantil, menaik tegas.

    Returns:
        Rata-rata CRPS atas seluruh jendela dan horizon.
    """
    return float(crps_elementwise(y_true, y_pred_quantiles, quantile_levels).mean())


def normalized_crps(
    y_true: Any, y_pred_quantiles: Any, quantile_levels: Any
) -> float:
    """Menghitung CRPS ternormalisasi: ``CRPS / mean(|y_true|)``.

    Normalisasi membuat angkanya tidak bersatuan sehingga dapat dibandingkan
    antar periode maupun antar aset dengan tingkat harga berbeda.

    Args:
        y_true: Nilai aktual ``(n_windows, H)``.
        y_pred_quantiles: Ramalan kuantil ``(n_windows, H, n_quantiles)``.
        quantile_levels: Daftar level kuantil, menaik tegas.

    Returns:
        CRPS dibagi rata-rata nilai absolut aktual.

    Raises:
        EvaluationError: Bila rata-rata ``|y_true|`` nol.
    """
    actual = _as_2d(y_true, "y_true")
    scale = float(np.abs(actual).mean())

    if np.isclose(scale, 0.0):
        raise EvaluationError(
            "Rata-rata |y_true| nol sehingga CRPS ternormalisasi tidak terdefinisi."
        )

    return float(crps_approx(y_true, y_pred_quantiles, quantile_levels) / scale)


def coverage(y_true: Any, lower: Any, upper: Any) -> float:
    """Menghitung cakupan empiris sebuah interval prediksi.

    Batas dihitung **inklusif** (``lower <= y <= upper``). Interval 80% yang
    terkalibrasi baik akan menghasilkan nilai mendekati 0.80, dan interval 95%
    mendekati 0.95.

    Args:
        y_true: Nilai aktual ``(n_windows, H)``.
        lower: Batas bawah interval ``(n_windows, H)``.
        upper: Batas atas interval ``(n_windows, H)``.

    Returns:
        Proporsi observasi yang jatuh di dalam interval, dalam [0, 1].

    Raises:
        EvaluationError: Bila ada batas bawah melebihi batas atas.
    """
    actual = _as_2d(y_true, "y_true")
    lower_bound = _as_2d(lower, "lower")
    upper_bound = _as_2d(upper, "upper")

    _check_same_shape(actual, lower_bound, "y_true", "lower")
    _check_same_shape(actual, upper_bound, "y_true", "upper")

    if (lower_bound > upper_bound).any():
        raise EvaluationError(
            "Terdapat batas bawah yang melebihi batas atas (interval terbalik)."
        )

    inside = (actual >= lower_bound) & (actual <= upper_bound)
    return float(inside.mean())


def mean_interval_width(lower: Any, upper: Any) -> float:
    """Menghitung lebar rata-rata interval prediksi.

    Wajib dilaporkan berdampingan dengan :func:`coverage`: cakupan tinggi yang
    dicapai hanya karena intervalnya kelewat lebar bukanlah kalibrasi yang baik,
    dan hanya perbandingan kedua angka ini yang dapat mengungkapnya.

    Args:
        lower: Batas bawah interval ``(n_windows, H)``.
        upper: Batas atas interval ``(n_windows, H)``.

    Returns:
        Rata-rata ``upper - lower``.

    Raises:
        EvaluationError: Bila ada batas bawah melebihi batas atas.
    """
    lower_bound = _as_2d(lower, "lower")
    upper_bound = _as_2d(upper, "upper")
    _check_same_shape(lower_bound, upper_bound, "lower", "upper")

    if (lower_bound > upper_bound).any():
        raise EvaluationError(
            "Terdapat batas bawah yang melebihi batas atas (interval terbalik)."
        )

    return float((upper_bound - lower_bound).mean())


def quantile_crossing_rate(y_pred_quantiles: Any, quantile_levels: Any) -> float:
    """Menghitung proporsi terjadinya *quantile crossing*.

    Pelanggaran dihitung pada tiap pasangan kuantil **berurutan**: untuk
    ``tau_i < tau_j`` yang bersebelahan, ramalan seharusnya memenuhi
    ``q_i <= q_j``. Penyebutnya adalah
    ``n_windows * H * (n_quantiles - 1)``, yakni seluruh kombinasi
    (timestep x pasangan kuantil berurutan).

    Nilai > 0 menandakan sebaran prediktif tidak konsisten secara internal —
    cacat yang perlu dilaporkan meski metrik lain terlihat baik.

    Args:
        y_pred_quantiles: Ramalan kuantil ``(n_windows, H, n_quantiles)``.
        quantile_levels: Daftar level kuantil, menaik tegas.

    Returns:
        Proporsi pelanggaran dalam [0, 1].

    Raises:
        EvaluationError: Bila hanya tersedia satu level kuantil.
    """
    predicted = _as_3d(y_pred_quantiles, "y_pred_quantiles")
    levels = _validate_quantile_levels(quantile_levels, predicted.shape[2])

    if levels.size < 2:
        raise EvaluationError(
            "Dibutuhkan minimal dua level kuantil untuk menghitung crossing rate."
        )

    # levels sudah dipastikan menaik, jadi selisih negatif = pelanggaran
    violations = np.diff(predicted, axis=2) < 0.0
    return float(violations.mean())


# =============================================================================
# Uji signifikansi
# =============================================================================


def diebold_mariano(
    errors_a: Any, errors_b: Any, h: int, alpha: float = 0.05
) -> dict[str, Any]:
    """Menguji beda akurasi dua ramalan dengan uji Diebold-Mariano.

    Masukannya adalah **rugi** (*loss*) tiap ramalan, bukan galat bertanda —
    umumnya galat absolut atau galat kuadrat. Selisih rugi
    ``d_t = L_a,t - L_b,t`` diuji terhadap H0: ``E[d_t] = 0`` (kedua ramalan
    sama akuratnya).

    Karena jendela *rolling origin* saling tumpang tindih, ``d_t`` pasti
    berautokorelasi. Ragamnya karena itu diestimasi dengan HAC Newey-West
    memakai bobot Bartlett dan lag ``h - 1`` (keputusan pada CLAUDE.md)::

        var_hac = (1/n) * [gamma_0 + 2 * sum_{k=1..L} (1 - k/(L+1)) * gamma_k]

    Bila masukan berbentuk ``(n_windows, H)``, rugi dirata-ratakan pada sumbu
    horizon lebih dulu sehingga deret yang diuji berindeks origin — inilah
    sumbu tempat tumpang tindih terjadi.

    Statistik negatif berarti ramalan A memiliki rugi lebih kecil (A lebih
    baik); statistik positif berarti B lebih baik.

    Args:
        errors_a: Rugi ramalan A, ``(n_windows, H)`` atau ``(n_windows,)``.
        errors_b: Rugi ramalan B, bentuk sama dengan ``errors_a``.
        h: Panjang horizon; lag HAC dipakai sebesar ``h - 1``.
        alpha: Taraf nyata untuk penanda ``significant``.

    Returns:
        Dictionary berisi ``mean_loss_differential``, ``statistic``,
        ``p_value``, ``hac_lag``, ``n_obs``, ``significant``, dan ``better``.

    Raises:
        EvaluationError: Bila ``h`` < 1, bentuk kedua masukan berbeda, jumlah
            observasi kurang dari dua, atau lag HAC melebihi ``n - 1``.
    """
    if h < 1:
        raise EvaluationError(f"h harus >= 1, diterima {h}.")

    loss_a = np.asarray(errors_a, dtype=float)
    loss_b = np.asarray(errors_b, dtype=float)

    if loss_a.shape != loss_b.shape:
        raise EvaluationError(
            f"Bentuk errors_a {loss_a.shape} tidak sama dengan "
            f"errors_b {loss_b.shape}."
        )
    if not (np.isfinite(loss_a).all() and np.isfinite(loss_b).all()):
        raise EvaluationError("Masukan memuat NaN atau nilai tak berhingga.")

    differential = loss_a - loss_b
    if differential.ndim == 2:
        # Rata-ratakan pada horizon: tumpang tindih terjadi antar-origin
        differential = differential.mean(axis=1)
    elif differential.ndim != 1:
        raise EvaluationError(
            f"errors_a/errors_b harus 1-D atau 2-D, diterima {loss_a.shape}."
        )

    n_obs = int(differential.size)
    if n_obs < 2:
        raise EvaluationError(
            f"Dibutuhkan minimal 2 observasi untuk uji DM, diterima {n_obs}."
        )

    hac_lag = h - 1
    if hac_lag > n_obs - 1:
        raise EvaluationError(
            f"Lag HAC ({hac_lag}) melebihi n_obs - 1 ({n_obs - 1}); "
            f"jumlah jendela terlalu sedikit untuk horizon sepanjang ini."
        )

    mean_differential = float(differential.mean())
    centered = differential - mean_differential

    # gamma_0 + 2 * sum bobot Bartlett * gamma_k
    long_run_variance = float(np.mean(centered**2))
    for lag in range(1, hac_lag + 1):
        autocovariance = float(np.mean(centered[lag:] * centered[:-lag]))
        bartlett_weight = 1.0 - lag / (hac_lag + 1.0)
        long_run_variance += 2.0 * bartlett_weight * autocovariance

    variance_of_mean = long_run_variance / n_obs

    if variance_of_mean <= 0.0:
        # Terjadi bila seluruh selisih rugi identik (mis. dua ramalan sama persis)
        statistic = 0.0 if np.isclose(mean_differential, 0.0) else float("inf")
        p_value = 1.0 if np.isclose(mean_differential, 0.0) else 0.0
    else:
        statistic = mean_differential / float(np.sqrt(variance_of_mean))
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(statistic))))

    if np.isclose(mean_differential, 0.0):
        better = "seri"
    else:
        better = "a" if mean_differential < 0 else "b"

    return {
        "mean_loss_differential": mean_differential,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "hac_lag": int(hac_lag),
        "n_obs": n_obs,
        "alpha": float(alpha),
        "significant": bool(p_value < alpha),
        "better": better,
    }


# =============================================================================
# Agregasi
# =============================================================================


def _summarize(values: np.ndarray) -> tuple[float, float]:
    """Meringkas sebuah array menjadi rata-rata dan simpangan baku sampel.

    Args:
        values: Array nilai yang akan diringkas.

    Returns:
        Tuple ``(mean, std)``; ``std`` bernilai 0 bila hanya ada satu nilai.
    """
    flat = np.asarray(values, dtype=float).ravel()
    std = float(flat.std(ddof=1)) if flat.size > 1 else 0.0
    return float(flat.mean()), std


def aggregate_metrics(
    y_true: Any,
    y_pred_quantiles: Any,
    quantile_levels: Any,
    y_naive: Any | None = None,
    coverage_levels: Any | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Menghitung seluruh metrik, per horizon dan secara keseluruhan.

    Dua tingkat agregasi dilaporkan:

    * ``overall`` — tiap jendela lebih dulu diringkas menjadi satu nilai
      (dirata-ratakan atas H langkahnya), lalu dilaporkan rata-rata dan
      **simpangan baku antar jendela**. Simpangan baku inilah yang menunjukkan
      seberapa konsisten performa model dari satu origin ke origin lain.
    * ``per_horizon`` — satu entri untuk tiap h = 1..H, memperlihatkan
      pemburukan akurasi seiring bertambah jauhnya horizon. Pada h tetap, tiap
      jendela hanya menyumbang satu nilai, sehingga simpangan baku yang
      dilaporkan adalah sebaran nilai tersebut antar jendela. RMSE per horizon
      tidak disertai simpangan baku karena merupakan agregat tak linear atas
      jendela; sebarannya sudah diwakili ``mae_std_windows``.

    Args:
        y_true: Nilai aktual ``(n_windows, H)``.
        y_pred_quantiles: Ramalan kuantil ``(n_windows, H, n_quantiles)``.
        quantile_levels: Daftar level kuantil, menaik tegas.
        y_naive: Ramalan titik baseline ``(n_windows, H)`` untuk MASE. Bila
            ``None``, MASE dilewati.
        coverage_levels: Daftar tingkat cakupan, mis. ``[0.80, 0.95]``. Bila
            ``None``, diambil dari ``evaluation.coverage_levels`` pada config.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.

    Returns:
        Dictionary metrik lengkap beserta metadata bentuk data.
    """
    if config is None:
        config = load_config()

    actual = _as_2d(y_true, "y_true")
    predicted = _as_3d(y_pred_quantiles, "y_pred_quantiles")
    levels = _validate_quantile_levels(quantile_levels, predicted.shape[2])

    if actual.shape != predicted.shape[:2]:
        raise EvaluationError(
            f"Dua dimensi pertama y_pred_quantiles {predicted.shape[:2]} tidak "
            f"sama dengan bentuk y_true {actual.shape}."
        )

    if coverage_levels is None:
        coverage_levels = config["evaluation"]["coverage_levels"]

    point_quantile = config["evaluation"]["point_quantile"]
    median_index = find_quantile_index(levels, point_quantile)
    point_forecast = predicted[:, :, median_index]

    n_windows, horizon = actual.shape

    # --- Rugi per elemen: dasar seluruh agregasi ---
    element_absolute = np.abs(actual - point_forecast)
    element_squared = (actual - point_forecast) ** 2
    element_percentage = absolute_percentage_errors(actual, point_forecast)
    element_crps = crps_elementwise(actual, predicted, levels)

    scale = float(np.abs(actual).mean())

    naive_forecast = None
    element_naive_absolute = None
    if y_naive is not None:
        naive_forecast = _as_2d(y_naive, "y_naive")
        _check_same_shape(actual, naive_forecast, "y_true", "y_naive")
        element_naive_absolute = np.abs(actual - naive_forecast)

    # --- Interval prediksi untuk tiap tingkat cakupan ---
    intervals: dict[str, dict[str, np.ndarray]] = {}
    for level in coverage_levels:
        lower_index, upper_index = interval_indices(levels, level)
        intervals[f"{level:g}"] = {
            "lower": predicted[:, :, lower_index],
            "upper": predicted[:, :, upper_index],
        }

    # ------------------------------------------------------------------ overall
    # Tiap jendela diringkas dulu -> simpangan bakunya bermakna "antar jendela"
    per_window_mae = element_absolute.mean(axis=1)
    per_window_rmse = np.sqrt(element_squared.mean(axis=1))
    per_window_mape = element_percentage.mean(axis=1)
    per_window_crps = element_crps.mean(axis=1)

    mae_mean, mae_std = _summarize(per_window_mae)
    rmse_mean, rmse_std = _summarize(per_window_rmse)
    mape_mean, mape_std = _summarize(per_window_mape)
    crps_mean, crps_std = _summarize(per_window_crps)

    overall: dict[str, Any] = {
        "mae": float(element_absolute.mean()),
        "mae_std_windows": mae_std,
        "rmse": float(np.sqrt(element_squared.mean())),
        "rmse_mean_of_windows": rmse_mean,
        "rmse_std_windows": rmse_std,
        "mape": float(element_percentage.mean()),
        "mape_std_windows": mape_std,
        "crps": float(element_crps.mean()),
        "crps_std_windows": crps_std,
        "normalized_crps": float(element_crps.mean() / scale),
        "quantile_crossing_rate": quantile_crossing_rate(predicted, levels),
        "coverage": {},
        "interval_width": {},
        "interval_width_std_windows": {},
    }
    # mae_mean identik dengan overall["mae"] karena seluruh jendela berbobot sama
    assert np.isclose(mae_mean, overall["mae"])

    for key, bounds in intervals.items():
        overall["coverage"][key] = coverage(actual, bounds["lower"], bounds["upper"])
        width = bounds["upper"] - bounds["lower"]
        overall["interval_width"][key] = float(width.mean())
        overall["interval_width_std_windows"][key] = _summarize(width.mean(axis=1))[1]

    if element_naive_absolute is not None:
        overall["mase"] = mase(actual, point_forecast, naive_forecast)
        overall["mae_naive"] = float(element_naive_absolute.mean())

    # -------------------------------------------------------------- per horizon
    per_horizon: list[dict[str, Any]] = []
    for step in range(horizon):
        step_absolute = element_absolute[:, step]
        step_percentage = element_percentage[:, step]
        step_crps = element_crps[:, step]

        entry: dict[str, Any] = {
            "h": step + 1,
            "n_windows": int(n_windows),
            "mae": float(step_absolute.mean()),
            "mae_std_windows": _summarize(step_absolute)[1],
            "rmse": float(np.sqrt(element_squared[:, step].mean())),
            "mape": float(step_percentage.mean()),
            "mape_std_windows": _summarize(step_percentage)[1],
            "crps": float(step_crps.mean()),
            "crps_std_windows": _summarize(step_crps)[1],
            "normalized_crps": float(
                step_crps.mean() / float(np.abs(actual[:, step]).mean())
            ),
            "quantile_crossing_rate": quantile_crossing_rate(
                predicted[:, step : step + 1, :], levels
            ),
            "coverage": {},
            "interval_width": {},
        }

        for key, bounds in intervals.items():
            lower_step = bounds["lower"][:, step : step + 1]
            upper_step = bounds["upper"][:, step : step + 1]
            entry["coverage"][key] = coverage(
                actual[:, step : step + 1], lower_step, upper_step
            )
            entry["interval_width"][key] = mean_interval_width(lower_step, upper_step)

        if element_naive_absolute is not None:
            naive_step_mae = float(element_naive_absolute[:, step].mean())
            entry["mae_naive"] = naive_step_mae
            entry["mase"] = (
                float(step_absolute.mean() / naive_step_mae)
                if not np.isclose(naive_step_mae, 0.0)
                else None
            )

        per_horizon.append(entry)

    return {
        "n_windows": int(n_windows),
        "horizon": int(horizon),
        "quantile_levels": [float(level) for level in levels],
        "point_quantile": float(point_quantile),
        "coverage_levels": [float(level) for level in coverage_levels],
        "target_scale_mean_abs": scale,
        "overall": overall,
        "per_horizon": per_horizon,
    }


# =============================================================================
# Titik masuk mandiri
# =============================================================================


def main() -> dict[str, Any]:
    """Titik masuk eksekusi mandiri: mengevaluasi baseline pada periode validasi.

    Menjalankan protokol *rolling origin* (keputusan D4) memakai baseline naive
    persistence pada data **validasi**, lalu melaporkan seluruh metrik. Data uji
    tidak disentuh sama sekali. Tujuannya menguji modul evaluasi ini pada data
    sungguhan sebelum modul model dibangun, sekaligus menghasilkan angka
    pembanding baseline pada periode validasi.

    Returns:
        Dictionary metrik lengkap yang juga disimpan ke JSON.
    """
    import pandas as pd

    from src.baseline import naive_persistence, naive_persistence_quantiles
    from src.data_collection import DATE_INDEX_NAME

    config = load_config()
    logger = setup_logger(
        "evaluation", log_dir=config.get("paths", {}).get("logs_dir", "logs")
    )

    target_column = get_target_column(config)
    horizon = config["model"]["prediction_length"]
    quantile_levels = config["model"]["quantile_levels"]
    stride = config["evaluation"]["stride"]

    logger.info("=" * 78)
    logger.info("EVALUASI — baseline naive persistence pada periode VALIDASI")
    logger.info("=" * 78)

    processed_dir = resolve_path(config["data"]["processed_dir"])
    frames = {}
    for name in ("train", "val"):
        frame = pd.read_csv(
            processed_dir / f"{name}.csv",
            index_col=DATE_INDEX_NAME,
            parse_dates=[DATE_INDEX_NAME],
        )
        frames[name] = frame.sort_index()
        logger.info("Memuat %s.csv: %d baris", name, len(frame))
    logger.info("Data UJI tidak dibaca sama sekali (aturan no. 2).")

    series = pd.concat([frames["train"][target_column], frames["val"][target_column]])
    n_train = len(frames["train"])
    values = series.to_numpy(dtype=float)

    # Jendela rolling origin: origin pertama = akhir data latih, sasaran selalu
    # berada di dalam periode validasi.
    origins = list(range(n_train, len(values) - horizon + 1, stride))
    if not origins:
        raise EvaluationError(
            "Periode validasi terlalu pendek untuk membentuk satu jendela pun."
        )

    y_true = np.empty((len(origins), horizon), dtype=float)
    y_naive = np.empty((len(origins), horizon), dtype=float)
    y_quantiles = np.empty((len(origins), horizon, len(quantile_levels)), dtype=float)

    for index, origin in enumerate(origins):
        context = values[:origin]  # expanding window, hanya masa lalu
        y_true[index] = values[origin : origin + horizon]
        y_naive[index] = naive_persistence(context, horizon)
        y_quantiles[index] = naive_persistence_quantiles(
            context, horizon, quantile_levels, config=config
        )

    logger.info(
        "Jendela terbentuk: %d (H = %d, stride = %d)", len(origins), horizon, stride
    )
    # `origin` adalah indeks langkah PERTAMA yang diramalkan; konteks berakhir
    # satu langkah sebelumnya. Dinyatakan eksplisit agar tidak rancu saat dikutip.
    for label, origin in (("pertama", origins[0]), ("terakhir", origins[-1])):
        logger.info(
            "Jendela %-8s: konteks s.d. %s  ->  target %s s.d. %s",
            label,
            series.index[origin - 1].date(),
            series.index[origin].date(),
            series.index[origin + horizon - 1].date(),
        )

    metrics = aggregate_metrics(
        y_true, y_quantiles, quantile_levels, y_naive=y_naive, config=config
    )
    metrics["protocol"] = {
        "split_evaluated": "val",
        "stride": int(stride),
        "n_windows": len(origins),
        "context_end_first_window": str(series.index[origins[0] - 1].date()),
        "target_first": str(series.index[origins[0]].date()),
        "target_last": str(series.index[origins[-1] + horizon - 1].date()),
        "context_type": "expanding (seluruh riwayat sampai origin)",
        "model": "naive_persistence",
    }

    _print_summary(metrics, logger)

    output_path = save_json(
        metrics,
        resolve_path(config["paths"]["metrics_dir"]) / "baseline_validation_metrics.json",
    )
    logger.info("Metrik baseline tersimpan di %s", output_path)

    return metrics


def _print_summary(metrics: dict[str, Any], logger: logging.Logger) -> None:
    """Mencetak ringkasan metrik ke log.

    Args:
        metrics: Keluaran :func:`aggregate_metrics`.
        logger: Logger tujuan.
    """
    overall = metrics["overall"]

    logger.info("-" * 78)
    logger.info("METRIK KESELURUHAN (rata-rata +/- simpangan baku antar jendela)")
    logger.info("  MAE              : %.4f +/- %.4f", overall["mae"], overall["mae_std_windows"])
    logger.info("  RMSE             : %.4f +/- %.4f", overall["rmse"], overall["rmse_std_windows"])
    logger.info("  MAPE             : %.4f%% +/- %.4f", overall["mape"], overall["mape_std_windows"])
    if "mase" in overall:
        logger.info("  MASE             : %.4f  (baseline vs dirinya sendiri = 1)", overall["mase"])
    logger.info("  CRPS             : %.4f +/- %.4f", overall["crps"], overall["crps_std_windows"])
    logger.info("  Normalized CRPS  : %.6f", overall["normalized_crps"])
    logger.info("  Crossing rate    : %.6f", overall["quantile_crossing_rate"])
    for key, value in overall["coverage"].items():
        logger.info(
            "  Coverage %-5s   : %.4f  (target %s, lebar rata-rata %.4f)",
            key,
            value,
            key,
            overall["interval_width"][key],
        )

    logger.info("-" * 78)
    logger.info("METRIK PER HORIZON")
    logger.info(
        "%-4s %10s %10s %10s %10s %10s %10s",
        "h", "MAE", "RMSE", "MAPE%", "CRPS", "cov80", "cov95",
    )
    for entry in metrics["per_horizon"]:
        logger.info(
            "%-4d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f",
            entry["h"],
            entry["mae"],
            entry["rmse"],
            entry["mape"],
            entry["crps"],
            entry["coverage"].get("0.8", float("nan")),
            entry["coverage"].get("0.95", float("nan")),
        )
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
