"""Baseline naive persistence untuk peramalan harga Perak.

Baseline ini menjalankan dua peran dalam penelitian:

1. **Pembanding akurasi** — model Chronos-2 (zero-shot maupun fine-tuned) harus
   mengalahkannya agar dapat dikatakan bermanfaat.
2. **Penyebut MASE** — sesuai spesifikasi metrik pada CLAUDE.md.

Dua varian disediakan:

* :func:`naive_persistence` — prediksi **titik**: nilai terakhir yang diketahui
  diulang H langkah ke depan.
* :func:`naive_persistence_quantiles` — prediksi **probabilistik**: mengikuti
  model *random walk*, ``log P_{t+h} = log P_t + sum(r)`` dengan
  ``r ~ N(0, sigma^2)``, sehingga sebaran pada horizon ke-h memiliki simpangan
  baku ``sigma * sqrt(h)``. Nilai sigma diestimasi dari log-return pada jendela
  terakhir **konteks** — tidak pernah dari masa depan, sehingga bebas kebocoran.

Karena kuantil 0.5 dari lognormal berpusat di ``log P_t`` bernilai tepat
``P_t``, prediksi titik dan median prediksi probabilistik selalu konsisten.

Cara menjalankan mandiri:
    python -m src.baseline
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.preprocessing import get_target_column
from src.utils import load_config, setup_logger


class BaselineError(ValueError):
    """Kegagalan pada pembentukan ramalan baseline."""


def _as_context_array(context_series: Any) -> np.ndarray:
    """Mengubah konteks menjadi array 1-D bersih dan memvalidasinya.

    Args:
        context_series: Riwayat harga (array-like, list, atau ``pd.Series``).

    Returns:
        Array 1-D berisi nilai konteks.

    Raises:
        BaselineError: Bila konteks kosong, bukan 1-D, atau memuat nilai tidak
            berhingga.
    """
    if isinstance(context_series, pd.Series):
        values = context_series.to_numpy(dtype=float)
    else:
        values = np.asarray(context_series, dtype=float)

    if values.ndim != 1:
        raise BaselineError(f"Konteks harus 1-D, diterima bentuk {values.shape}.")
    if values.size == 0:
        raise BaselineError("Konteks kosong, ramalan baseline tidak dapat dibentuk.")
    if not np.isfinite(values).all():
        raise BaselineError("Konteks memuat NaN atau nilai tak berhingga.")

    return values


def naive_persistence(context_series: Any, horizon: int) -> np.ndarray:
    """Membentuk ramalan titik naive persistence.

    Nilai terakhir yang diketahui diulang sebanyak ``horizon`` langkah. Inilah
    ramalan optimal di bawah hipotesis *random walk* tanpa drift, dan menjadi
    tolok ukur minimal yang harus dilewati model.

    Args:
        context_series: Riwayat harga sampai titik origin (1-D).
        horizon: Banyaknya langkah ke depan yang diramalkan (H).

    Returns:
        Array bentuk ``(horizon,)`` berisi nilai terakhir yang diulang.

    Raises:
        BaselineError: Bila konteks tidak valid atau ``horizon`` < 1.
    """
    values = _as_context_array(context_series)

    if horizon < 1:
        raise BaselineError(f"horizon harus >= 1, diterima {horizon}.")

    return np.full(horizon, values[-1], dtype=float)


def estimate_random_walk_sigma(
    context_series: Any,
    volatility_window: int,
    min_volatility_window: int = 2,
) -> float:
    """Mengestimasi simpangan baku log-return harian dari ekor konteks.

    Hanya ``volatility_window`` log-return terakhir yang dipakai, agar estimasi
    volatilitas mencerminkan rezim pasar terkini dan tetap konsisten dengan
    protokol *rolling origin* (keputusan D4): setiap origin memakai informasi
    yang tersedia sampai origin tersebut saja.

    Args:
        context_series: Riwayat harga sampai titik origin (1-D, seluruhnya > 0).
        volatility_window: Banyaknya log-return terakhir yang dipakai.
        min_volatility_window: Jumlah log-return minimum yang wajib tersedia.

    Returns:
        Simpangan baku log-return (sigma harian).

    Raises:
        BaselineError: Bila ada harga <= 0, jendela tidak valid, log-return yang
            tersedia kurang dari ``min_volatility_window``, atau sigma nol.
    """
    values = _as_context_array(context_series)

    if volatility_window < 2:
        raise BaselineError(
            f"volatility_window harus >= 2, diterima {volatility_window}."
        )
    if (values <= 0).any():
        raise BaselineError("Terdapat harga <= 0, log-return tidak terdefinisi.")

    log_returns = np.diff(np.log(values))
    if log_returns.size < min_volatility_window:
        raise BaselineError(
            f"Konteks hanya menghasilkan {log_returns.size} log-return, "
            f"minimum yang dibutuhkan {min_volatility_window}."
        )

    window = log_returns[-volatility_window:]
    sigma = float(window.std(ddof=1))

    if not np.isfinite(sigma) or sigma <= 0.0:
        raise BaselineError(
            "Simpangan baku log-return nol atau tidak berhingga; "
            "interval baseline tidak dapat dibentuk."
        )

    return sigma


def naive_persistence_quantiles(
    context_series: Any,
    horizon: int,
    quantile_levels: list[float],
    volatility_window: int | None = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Membentuk ramalan probabilistik baseline berbasis *random walk*.

    Di bawah model ``log P_{t+h} = log P_t + sum_{i=1..h} r_i`` dengan
    ``r_i ~ N(0, sigma^2)`` yang saling bebas, sebaran ``log P_{t+h}`` adalah
    normal dengan rata-rata ``log P_t`` dan simpangan baku ``sigma * sqrt(h)``.
    Kuantil ke-tau karenanya:

        Q_tau(h) = P_t * exp(z_tau * sigma * sqrt(h))

    dengan ``z_tau`` kuantil normal baku. Lebar interval otomatis melebar
    sebanding akar horizon — perilaku yang wajar untuk harga aset dan
    memberi pembanding yang adil bagi interval keluaran Chronos-2.

    Tanpa drift, sehingga ``Q_0.5(h) = P_t`` untuk seluruh h dan median di sini
    selalu identik dengan keluaran :func:`naive_persistence`.

    Args:
        context_series: Riwayat harga sampai titik origin (1-D, seluruhnya > 0).
        horizon: Banyaknya langkah ke depan (H).
        quantile_levels: Level kuantil yang diminta, tiap nilai dalam (0, 1).
        volatility_window: Jendela estimasi sigma. Bila ``None``, diambil dari
            ``baseline.volatility_window`` pada konfigurasi.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.

    Returns:
        Array bentuk ``(horizon, len(quantile_levels))``. Kolom mengikuti urutan
        ``quantile_levels`` yang diberikan, dan karena transformasinya monoton
        naik, tidak akan pernah terjadi *quantile crossing*.

    Raises:
        BaselineError: Bila ``horizon`` < 1, daftar kuantil kosong, atau ada
            level kuantil di luar rentang (0, 1).
    """
    if config is None:
        config = load_config()

    baseline_config = config["baseline"]
    if volatility_window is None:
        volatility_window = baseline_config["volatility_window"]

    if horizon < 1:
        raise BaselineError(f"horizon harus >= 1, diterima {horizon}.")

    levels = np.asarray(quantile_levels, dtype=float)
    if levels.size == 0:
        raise BaselineError("quantile_levels kosong.")
    if ((levels <= 0.0) | (levels >= 1.0)).any():
        raise BaselineError(
            f"Seluruh level kuantil harus berada di (0, 1), diterima {quantile_levels}."
        )

    values = _as_context_array(context_series)
    last_value = values[-1]

    sigma = estimate_random_walk_sigma(
        values,
        volatility_window=volatility_window,
        min_volatility_window=baseline_config["min_volatility_window"],
    )

    # Simpangan baku kumulatif per horizon: sigma * sqrt(h), h = 1..H
    horizon_std = sigma * np.sqrt(np.arange(1, horizon + 1, dtype=float))
    z_scores = stats.norm.ppf(levels)

    # (horizon, 1) x (1, n_quantiles) -> (horizon, n_quantiles)
    return last_value * np.exp(horizon_std[:, None] * z_scores[None, :])


def main() -> dict[str, Any]:
    """Titik masuk eksekusi mandiri: mendemonstrasikan baseline pada data latih.

    Mengambil seluruh data latih sebagai konteks, lalu membentuk ramalan titik
    dan ramalan probabilistik H langkah ke depan. Data validasi maupun uji tidak
    disentuh sama sekali.

    Returns:
        Dictionary berisi ramalan titik, ramalan kuantil, dan sigma terestimasi.
    """
    config = load_config()
    logger = setup_logger(
        "baseline", log_dir=config.get("paths", {}).get("logs_dir", "logs")
    )

    from src.mutual_information import load_train_data  # impor lokal: hindari siklus

    target_column = get_target_column(config)
    horizon = config["model"]["prediction_length"]
    quantile_levels = config["model"]["quantile_levels"]
    volatility_window = config["baseline"]["volatility_window"]

    logger.info("=" * 78)
    logger.info("BASELINE naive persistence — demonstrasi pada data latih")
    logger.info("=" * 78)

    train = load_train_data(config=config, logger=logger)
    context = train[target_column]

    point_forecast = naive_persistence(context, horizon)
    quantile_forecast = naive_persistence_quantiles(
        context, horizon, quantile_levels, config=config
    )
    sigma = estimate_random_walk_sigma(
        context,
        volatility_window=volatility_window,
        min_volatility_window=config["baseline"]["min_volatility_window"],
    )

    logger.info("Origin (nilai terakhir konteks) : %.4f", context.iloc[-1])
    logger.info("Tanggal origin                  : %s", context.index[-1].date())
    logger.info("Horizon H                       : %d langkah", horizon)
    logger.info(
        "Sigma log-return (%d hari)      : %.6f  (~%.2f%% per hari)",
        volatility_window,
        sigma,
        sigma * 100,
    )

    median_index = quantile_levels.index(config["evaluation"]["point_quantile"])
    assert np.allclose(quantile_forecast[:, median_index], point_forecast), (
        "Median ramalan probabilistik wajib identik dengan ramalan titik."
    )
    logger.info("Median ramalan probabilistik identik dengan ramalan titik: OK")

    low_80 = quantile_levels.index(0.1)
    high_80 = quantile_levels.index(0.9)
    logger.info("-" * 78)
    logger.info("%-4s %12s %12s %12s %12s", "h", "titik", "q0.10", "q0.90", "lebar 80%")
    for step in range(horizon):
        logger.info(
            "%-4d %12.4f %12.4f %12.4f %12.4f",
            step + 1,
            point_forecast[step],
            quantile_forecast[step, low_80],
            quantile_forecast[step, high_80],
            quantile_forecast[step, high_80] - quantile_forecast[step, low_80],
        )
    logger.info("=" * 78)

    return {
        "origin_value": float(context.iloc[-1]),
        "origin_date": str(context.index[-1].date()),
        "sigma": sigma,
        "point_forecast": point_forecast,
        "quantile_forecast": quantile_forecast,
        "quantile_levels": quantile_levels,
    }


if __name__ == "__main__":
    main()
