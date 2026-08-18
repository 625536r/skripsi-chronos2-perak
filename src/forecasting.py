"""Peramalan Chronos-2: skema zero-shot dan skema fine-tuned.

Modul ini membungkus AutoGluon-TimeSeries dan menjalankan protokol *rolling
origin* (keputusan D4). Seluruh keputusan metodologis pada CLAUDE.md yang
menyentuh model diterapkan di sini:

D2 -- Kovariat PAST-ONLY
    ``TimeSeriesPredictor`` selalu dibentuk dengan ``known_covariates_names=[]``.
    Emas dan Dollar Index karenanya otomatis diperlakukan sebagai *past-only
    covariates*: model hanya boleh melihat nilainya sampai titik origin, tidak
    pernah nilai masa depannya. Nilainya juga tidak pernah diteruskan lewat
    argumen ``known_covariates`` pada ``predict()``.

D3 -- Peran data validasi
    Data latih dipakai untuk fine-tuning bobot; data validasi HANYA dipakai
    memilih hyperparameter fine-tuning; data uji hanya disentuh sekali pada
    pelaporan akhir.

D4 -- Rolling origin (expanding window)
    Periode evaluasi tidak diramalkan sekaligus. Untuk tiap origin, konteksnya
    adalah seluruh riwayat sampai origin (dibatasi ``max_context``), lalu model
    meramalkan H langkah ke depan. Zero-shot, fine-tuned, dan baseline WAJIB
    memakai daftar origin yang identik -- dijamin oleh
    :func:`build_origins`, yang dipanggil satu kali dan dipakai bersama.

D5 -- Level kuantil diambil apa adanya dari ``model.quantile_levels``.

Catatan penting soal indeks waktu
---------------------------------
Kalender bursa tidak reguler (akhir pekan dan hari libur), sehingga indeks
tanggal asli berfrekuensi tidak tetap dan ditolak AutoGluon. Karena Chronos-2
TIDAK memakai timestamp sebagai fitur -- timestamp hanya dipakai untuk menomori
langkah ramalan -- tanggal asli dipetakan 1:1 ke indeks sintetis berfrekuensi
tetap (``model.synthetic_index`` pada config). Dampaknya: satu langkah model
sama dengan satu hari perdagangan, persis definisi H pada keputusan D4.
Seluruh keluaran yang disimpan ke disk tetap memakai tanggal perdagangan asli.

Cara menjalankan mandiri (uji asap zero-shot pada periode VALIDASI):
    python -m src.forecasting
    python -m src.forecasting --schemes zeroshot --split val --horizon 10

Menjalankan seluruh skema pada periode uji (dipanggil oleh run_pipeline.py):
    python -m src.forecasting --schemes zeroshot finetuned naive --split test
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from src.baseline import naive_persistence, naive_persistence_quantiles
from src.data_collection import DATE_INDEX_NAME
from src.evaluation import crps_approx, find_quantile_index, normalized_crps
from src.preprocessing import get_covariate_columns, get_target_column
from src.utils import load_config, resolve_path, save_json, setup_logger

# Nama skema yang dikenali modul ini
SCHEME_ZEROSHOT = "zeroshot"
SCHEME_FINETUNED = "finetuned"
SCHEME_NAIVE = "naive"
ALL_SCHEMES = (SCHEME_ZEROSHOT, SCHEME_FINETUNED, SCHEME_NAIVE)

# Kunci model Chronos-2 pada dict `hyperparameters` AutoGluon
# (DIVERIFIKASI pada autogluon.timeseries 1.6.1: alias "Chronos2" -> Chronos2Model)
CHRONOS2_KEY = "Chronos2"

# Nilai `fine_tune_mode` yang benar-benar didukung chronos 2.3.1
VALID_FINE_TUNE_MODES = ("lora", "full")


class ForecastingError(RuntimeError):
    """Kegagalan pada pembentukan predictor atau proses peramalan."""


# =============================================================================
# Penyiapan data untuk model
# =============================================================================


def model_columns(config: dict[str, Any]) -> list[str]:
    """Menyusun daftar kolom yang diberikan ke model: target lalu kovariat.

    Kolom flag ``*_ffilled`` sengaja tidak disertakan karena sifatnya metadata
    pelaporan, bukan variabel model.

    Args:
        config: Konfigurasi project.

    Returns:
        Daftar nama kolom, dimulai dari kolom target.
    """
    return [get_target_column(config), *get_covariate_columns(config)]


def synthetic_timestamps(n_rows: int, config: dict[str, Any]) -> pd.DatetimeIndex:
    """Membentuk indeks waktu sintetis berfrekuensi tetap sepanjang ``n_rows``.

    Lihat catatan indeks waktu pada docstring modul: pemetaan ini murni teknis
    agar AutoGluon menerima data, dan tidak mengubah informasi apa pun yang
    dilihat model.

    Args:
        n_rows: Banyaknya baris (hari perdagangan) yang perlu diberi timestamp.
        config: Konfigurasi project.

    Returns:
        ``DatetimeIndex`` berfrekuensi tetap sepanjang ``n_rows``.
    """
    index_config = config["model"]["synthetic_index"]
    return pd.date_range(
        start=index_config["start"], periods=n_rows, freq=index_config["freq"]
    )


def to_model_tsdf(
    df: pd.DataFrame,
    config: dict[str, Any],
    end_position: int | None = None,
) -> TimeSeriesDataFrame:
    """Mengubah potongan dataframe harga menjadi ``TimeSeriesDataFrame`` siap pakai.

    Berbeda dengan :func:`src.preprocessing.to_timeseries_dataframe` yang
    mempertahankan tanggal asli (dan karenanya menghasilkan ``freq=None``),
    fungsi ini memasang indeks sintetis berfrekuensi tetap sehingga dapat
    diterima AutoGluon.

    ``end_position`` menjaga agar potongan konteks pada protokol rolling origin
    tetap menempati posisi timestamp yang sama dengan posisinya di deret penuh.
    Dengan begitu timestamp ramalan H langkah ke depan selalu sinkron dengan
    posisi aktual yang dibandingkan.

    Args:
        df: Potongan dataframe dengan index tanggal asli.
        config: Konfigurasi project.
        end_position: Posisi (0-based, eksklusif) baris terakhir potongan pada
            deret penuh. Bila ``None``, potongan dianggap dimulai dari posisi 0.

    Returns:
        ``TimeSeriesDataFrame`` berisi satu item dengan frekuensi tetap.

    Raises:
        ForecastingError: Bila ada kolom model yang hilang atau potongan kosong.
    """
    columns = model_columns(config)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ForecastingError(
            f"Kolom berikut tidak ada pada dataframe: {missing}. "
            f"Kolom tersedia: {list(df.columns)}"
        )
    if df.empty:
        raise ForecastingError("Potongan dataframe kosong, TSDF tidak dapat dibentuk.")

    n_rows = len(df)
    end = n_rows if end_position is None else end_position
    if end < n_rows:
        raise ForecastingError(
            f"end_position ({end}) tidak boleh lebih kecil dari panjang "
            f"potongan ({n_rows})."
        )

    # Timestamp sintetis untuk seluruh deret sampai `end`, lalu diambil ekornya
    # sepanjang potongan -> posisi potongan pada sumbu waktu terjaga.
    timestamps = synthetic_timestamps(end, config)[end - n_rows : end]

    flat = df[columns].reset_index(drop=True)
    flat[TimeSeriesDataFrame.TIMESTAMP] = timestamps
    flat[TimeSeriesDataFrame.ITEMID] = config["model"]["item_id"]

    return TimeSeriesDataFrame.from_data_frame(
        flat,
        id_column=TimeSeriesDataFrame.ITEMID,
        timestamp_column=TimeSeriesDataFrame.TIMESTAMP,
    )


def load_processed_frames(
    names: tuple[str, ...] = ("train", "val", "test"),
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame]:
    """Memuat bagian-bagian data hasil preprocessing dari ``data/processed/``.

    Hanya berkas yang benar-benar diminta lewat ``names`` yang dibaca, sehingga
    pemanggil yang tidak boleh menyentuh data uji (mis. tahap tuning) memang
    tidak akan pernah membukanya.

    Args:
        names: Nama bagian yang dimuat, mis. ``("train", "val")``.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dictionary ``{nama: DataFrame}`` dengan index tanggal terurut menaik.

    Raises:
        FileNotFoundError: Bila salah satu berkas belum ada.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("forecasting")

    processed_dir = resolve_path(config["data"]["processed_dir"])
    frames: dict[str, pd.DataFrame] = {}

    for name in names:
        path = processed_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Berkas '{path}' tidak ditemukan. "
                f"Jalankan 'python -m src.preprocessing' terlebih dahulu."
            )
        frame = pd.read_csv(
            path, index_col=DATE_INDEX_NAME, parse_dates=[DATE_INDEX_NAME]
        )
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        frame.index.name = DATE_INDEX_NAME
        frames[name] = frame.sort_index()
        logger.info(
            "Memuat %-5s: %d baris (%s s.d. %s)",
            name,
            len(frame),
            frame.index[0].date(),
            frame.index[-1].date(),
        )

    return frames


# =============================================================================
# Pembentukan predictor
# =============================================================================


def _build_predictor(
    config: dict[str, Any],
    horizon: int,
    path: str,
    logger: logging.Logger,
) -> TimeSeriesPredictor:
    """Membentuk ``TimeSeriesPredictor`` kosong dengan setelan sesuai config.

    Titik penegakan keputusan D2 ada di sini: ``known_covariates_names`` diambil
    dari config dan wajib kosong.

    Args:
        config: Konfigurasi project.
        horizon: Panjang horizon H.
        path: Direktori penyimpanan artefak predictor.
        logger: Logger yang dipakai.

    Returns:
        Objek ``TimeSeriesPredictor`` yang belum di-fit.

    Raises:
        ForecastingError: Bila ``known_covariates_names`` tidak kosong.
    """
    model_config = config["model"]
    known_covariates = model_config["known_covariates_names"]

    if known_covariates:
        raise ForecastingError(
            "known_covariates_names WAJIB kosong (keputusan D2): nilai Emas dan "
            "Dollar Index di masa depan tidak diketahui saat peramalan dilakukan. "
            f"Diterima: {known_covariates}. Bila ini memang skenario ablasi, "
            "jalankan terpisah dan beri label 'oracle / ex-post'."
        )

    logger.info(
        "Membentuk TimeSeriesPredictor: target=%s, H=%d, known_covariates=%s (D2), "
        "eval_metric=%s",
        get_target_column(config),
        horizon,
        known_covariates,
        config["finetune"]["selection_metric"],
    )

    return TimeSeriesPredictor(
        target=get_target_column(config),
        known_covariates_names=known_covariates,
        prediction_length=horizon,
        quantile_levels=model_config["quantile_levels"],
        eval_metric=config["finetune"]["selection_metric"],
        path=path,
        cache_predictions=False,
        verbosity=1,
    )


def build_zeroshot_predictor(
    config: dict[str, Any] | None = None,
    train_tsdf: TimeSeriesDataFrame | None = None,
    horizon: int | None = None,
    path: str | None = None,
    logger: logging.Logger | None = None,
) -> TimeSeriesPredictor:
    """Membentuk predictor Chronos-2 **zero-shot** (tanpa pelatihan tambahan).

    ``TimeSeriesPredictor.fit`` tetap harus dipanggil karena itulah satu-satunya
    cara AutoGluon mendaftarkan model. Namun pada skema ini tidak ada satu pun
    bobot yang diperbarui:

    * ``fine_tune`` dibiarkan pada nilai bawaannya (``False``);
    * ``skip_model_selection=True`` sehingga AutoGluon tidak memotong data latih
      menjadi jendela validasi internal dan tidak melakukan seleksi model;
    * ``enable_ensemble=False`` sehingga hanya ada satu model.

    Data latih tetap diberikan karena dari situlah AutoGluon menyimpulkan
    frekuensi, kolom target, dan kolom mana yang menjadi *past covariates*
    (keputusan D2). Isinya tidak dipakai untuk melatih apa pun.

    Args:
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        train_tsdf: Data latih dalam bentuk ``TimeSeriesDataFrame``. Bila
            ``None``, dimuat dari ``data/processed/train.csv``.
        horizon: Panjang horizon H. Bila ``None``, diambil dari
            ``model.prediction_length``.
        path: Direktori artefak predictor. Bila ``None``, dibentuk di bawah
            ``paths.models_dir``.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        ``TimeSeriesPredictor`` yang siap dipakai untuk :func:`rolling_forecast`.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("forecasting")
    if horizon is None:
        horizon = config["model"]["prediction_length"]
    if train_tsdf is None:
        train_frame = load_processed_frames(("train",), config=config, logger=logger)[
            "train"
        ]
        train_tsdf = to_model_tsdf(train_frame, config)
    if path is None:
        path = str(
            resolve_path(config["paths"]["models_dir"]) / f"chronos2_zeroshot_H{horizon}"
        )

    predictor = _build_predictor(config, horizon, path, logger)

    hyperparameters = {
        CHRONOS2_KEY: [
            {
                "ag_args": {"name_suffix": "ZeroShot"},
                "cross_learning": config["model"]["cross_learning"],
                "context_length": config["model"]["max_context"],
            }
        ]
    }

    logger.info("Skema ZERO-SHOT: hyperparameters=%s", hyperparameters)
    started_at = time.perf_counter()

    predictor.fit(
        train_data=train_tsdf,
        hyperparameters=hyperparameters,
        enable_ensemble=False,
        skip_model_selection=True,
        random_seed=config["seed"],
    )

    logger.info(
        "Predictor zero-shot siap dalam %.2f detik (tidak ada bobot yang dilatih). "
        "Model: %s",
        time.perf_counter() - started_at,
        predictor.model_names(),
    )

    return predictor


def build_finetuned_predictor(
    train_tsdf: TimeSeriesDataFrame,
    best_params: dict[str, Any],
    config: dict[str, Any] | None = None,
    horizon: int | None = None,
    path: str | None = None,
    logger: logging.Logger | None = None,
) -> TimeSeriesPredictor:
    """Membentuk predictor Chronos-2 **fine-tuned** memakai hyperparameter terpilih.

    Fine-tuning dilakukan HANYA pada ``train_tsdf`` (keputusan D3). Data validasi
    sudah habis perannya pada :func:`tune_finetune_hyperparams` dan tidak boleh
    ikut melatih bobot; data uji tidak disentuh sama sekali di sini.

    ``skip_model_selection=True`` dipakai agar AutoGluon tidak memotong data
    latih menjadi jendela validasi internal — seluruh data latih dipakai untuk
    fine-tuning, dan penilaian dilakukan sendiri lewat :func:`rolling_forecast`.

    Args:
        train_tsdf: Data latih dalam bentuk ``TimeSeriesDataFrame``.
        best_params: Dictionary berisi ``fine_tune_lr``, ``fine_tune_steps``,
            dan ``fine_tune_mode`` hasil :func:`tune_finetune_hyperparams`.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        horizon: Panjang horizon H. Bila ``None``, diambil dari config.
        path: Direktori artefak predictor. Bila ``None``, dibentuk di bawah
            ``paths.models_dir``.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        ``TimeSeriesPredictor`` berisi satu model Chronos-2 yang sudah di-fine-tune.

    Raises:
        ForecastingError: Bila ``best_params`` tidak lengkap atau
            ``fine_tune_mode`` bukan nilai yang didukung.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("forecasting")
    if horizon is None:
        horizon = config["model"]["prediction_length"]
    if path is None:
        path = str(
            resolve_path(config["paths"]["models_dir"])
            / f"chronos2_finetuned_H{horizon}"
        )

    required = ("fine_tune_lr", "fine_tune_steps", "fine_tune_mode")
    missing = [key for key in required if key not in best_params]
    if missing:
        raise ForecastingError(
            f"best_params tidak lengkap, kunci yang hilang: {missing}."
        )
    if best_params["fine_tune_mode"] not in VALID_FINE_TUNE_MODES:
        raise ForecastingError(
            f"fine_tune_mode '{best_params['fine_tune_mode']}' tidak didukung; "
            f"nilai yang valid: {list(VALID_FINE_TUNE_MODES)}."
        )

    predictor = _build_predictor(config, horizon, path, logger)

    hyperparameters = {
        CHRONOS2_KEY: [
            {
                "ag_args": {"name_suffix": "FineTuned"},
                "fine_tune": True,
                "fine_tune_lr": float(best_params["fine_tune_lr"]),
                "fine_tune_steps": int(best_params["fine_tune_steps"]),
                "fine_tune_mode": str(best_params["fine_tune_mode"]),
                # Disesuaikan dengan ukuran data (lihat komentar pada config):
                # bawaan Chronos-2 ditujukan untuk korpus berisi ratusan deret.
                "fine_tune_batch_size": config["finetune"]["fine_tune_batch_size"],
                "fine_tune_context_length": config["finetune"][
                    "fine_tune_context_length"
                ],
                "cross_learning": config["model"]["cross_learning"],
                "context_length": config["model"]["max_context"],
            }
        ]
    }

    logger.info("Skema FINE-TUNED: hyperparameters=%s", hyperparameters)
    started_at = time.perf_counter()

    predictor.fit(
        train_data=train_tsdf,
        hyperparameters=hyperparameters,
        enable_ensemble=False,
        skip_model_selection=True,
        time_limit=config["finetune"]["time_limit"],
        random_seed=config["seed"],
    )

    logger.info(
        "Fine-tuning selesai dalam %.2f detik. Model: %s",
        time.perf_counter() - started_at,
        predictor.model_names(),
    )

    return predictor


# =============================================================================
# Rolling origin (keputusan D4)
# =============================================================================


def build_origins(
    n_total: int, eval_start_idx: int, horizon: int, stride: int
) -> list[int]:
    """Menyusun daftar posisi origin untuk protokol rolling origin.

    ``origin`` didefinisikan sebagai posisi (0-based) langkah **pertama yang
    diramalkan**; konteks berakhir tepat satu langkah sebelumnya. Sebuah origin
    hanya sah bila seluruh H langkah sasarannya masih tersedia di dalam data.

    Fungsi ini menjadi satu-satunya sumber daftar origin, sehingga seluruh skema
    (zero-shot, fine-tuned, baseline) dijamin dievaluasi pada jendela yang persis
    sama — syarat mutlak keputusan D4 dan prasyarat uji Diebold-Mariano.

    Args:
        n_total: Banyaknya baris pada deret penuh.
        eval_start_idx: Posisi baris pertama periode yang dievaluasi.
        horizon: Panjang horizon H.
        stride: Jarak antar origin.

    Returns:
        Daftar posisi origin terurut menaik.

    Raises:
        ForecastingError: Bila argumen tidak valid atau tidak ada satu pun
            jendela yang terbentuk.
    """
    if horizon < 1:
        raise ForecastingError(f"horizon harus >= 1, diterima {horizon}.")
    if stride < 1:
        raise ForecastingError(f"stride harus >= 1, diterima {stride}.")
    if not 0 < eval_start_idx <= n_total:
        raise ForecastingError(
            f"eval_start_idx harus di (0, {n_total}], diterima {eval_start_idx}."
        )

    origins = list(range(eval_start_idx, n_total - horizon + 1, stride))
    if not origins:
        raise ForecastingError(
            f"Tidak ada jendela yang terbentuk: periode evaluasi hanya "
            f"{n_total - eval_start_idx} baris, sedangkan H = {horizon}."
        )

    return origins


def rolling_forecast(
    predictor: TimeSeriesPredictor,
    full_df: pd.DataFrame,
    test_start_idx: int,
    H: int,
    stride: int,
    max_context: int,
    config: dict[str, Any] | None = None,
    origins: list[int] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Menjalankan peramalan rolling origin dengan konteks *expanding* (D4).

    Untuk setiap origin:

    1. konteks dipotong dari seluruh riwayat sampai origin, dibatasi
       ``max_context`` baris terakhir (expanding window yang dibatasi);
    2. ``predictor.predict(context_tsdf)`` dipanggil — tanpa argumen
       ``known_covariates``, sehingga nilai Emas dan DXY di masa depan tidak
       pernah bocor ke model (keputusan D2);
    3. kuantil ramalan dan nilai aktual H langkah ke depan dikumpulkan.

    Konteks selalu berhenti tepat sebelum origin, sehingga tidak mungkin ada
    kebocoran data ke belakang.

    Args:
        predictor: Predictor yang sudah di-fit (zero-shot maupun fine-tuned).
        full_df: Deret penuh (latih + validasi + uji) dengan index tanggal asli.
        test_start_idx: Posisi baris pertama periode yang dievaluasi.
        H: Panjang horizon.
        stride: Jarak antar origin.
        max_context: Batas panjang konteks (banyaknya baris terakhir).
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        origins: Daftar origin yang wajib dipakai. Bila ``None``, dibentuk oleh
            :func:`build_origins`. Argumen inilah yang dipakai untuk memaksa
            zero-shot dan fine-tuned memakai jendela yang identik.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dictionary berisi:

        * ``y_true``            : array ``(n_windows, H)``
        * ``y_pred_quantiles``  : array ``(n_windows, H, n_quantiles)``
        * ``y_pred``            : array ``(n_windows, H)`` (kuantil median)
        * ``origins``           : daftar posisi origin
        * ``origin_dates``      : tanggal asli langkah pertama tiap jendela
        * ``context_end_dates`` : tanggal asli baris terakhir konteks
        * ``target_dates``      : daftar tanggal asli sasaran per jendela
        * ``quantile_levels``   : level kuantil sesuai urutan kolom
        * ``elapsed_seconds``   : lama eksekusi seluruh jendela

    Raises:
        ForecastingError: Bila horizon predictor tidak sama dengan ``H``, atau
            keluaran ``predict`` tidak sepanjang H.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("forecasting")

    if predictor.prediction_length != H:
        raise ForecastingError(
            f"prediction_length predictor ({predictor.prediction_length}) tidak "
            f"sama dengan H yang diminta ({H}). Bentuk predictor terpisah untuk "
            f"tiap horizon."
        )
    if max_context < 1:
        raise ForecastingError(f"max_context harus >= 1, diterima {max_context}.")

    target_column = get_target_column(config)
    quantile_levels = config["model"]["quantile_levels"]
    quantile_columns = [str(level) for level in quantile_levels]

    values = full_df[target_column].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(full_df.index)
    n_total = len(full_df)

    if origins is None:
        origins = build_origins(n_total, test_start_idx, H, stride)

    n_windows = len(origins)
    y_true = np.empty((n_windows, H), dtype=float)
    y_pred_quantiles = np.empty((n_windows, H, len(quantile_levels)), dtype=float)

    logger.info(
        "Rolling origin: %d jendela, H=%d, stride=%d, max_context=%d, model=%s",
        n_windows,
        H,
        stride,
        max_context,
        predictor.model_names(),
    )

    # Menahan bobot model di memori: tanpa ini AutoGluon memuat ulang model dari
    # disk pada setiap panggilan predict, dan overhead itu berkali lipat karena
    # jumlah jendela banyak.
    predictor.persist()
    started_at = time.perf_counter()

    try:
        for window_index, origin in enumerate(origins):
            context_start = max(0, origin - max_context)
            context_frame = full_df.iloc[context_start:origin]
            context_tsdf = to_model_tsdf(context_frame, config, end_position=origin)

            # known_covariates sengaja TIDAK diteruskan (keputusan D2)
            prediction = predictor.predict(
                context_tsdf, random_seed=config["seed"]
            )

            if len(prediction) != H:
                raise ForecastingError(
                    f"predict() mengembalikan {len(prediction)} baris, "
                    f"diharapkan {H}."
                )

            y_true[window_index] = values[origin : origin + H]
            y_pred_quantiles[window_index] = prediction[quantile_columns].to_numpy(
                dtype=float
            )

            if (window_index + 1) % 25 == 0 or window_index + 1 == n_windows:
                elapsed = time.perf_counter() - started_at
                logger.info(
                    "  jendela %d/%d selesai (%.1f detik, %.2f detik/jendela)",
                    window_index + 1,
                    n_windows,
                    elapsed,
                    elapsed / (window_index + 1),
                )
    finally:
        predictor.unpersist()

    elapsed_seconds = time.perf_counter() - started_at
    median_index = find_quantile_index(quantile_levels, config["evaluation"]["point_quantile"])

    return {
        "y_true": y_true,
        "y_pred_quantiles": y_pred_quantiles,
        "y_pred": y_pred_quantiles[:, :, median_index],
        "origins": list(origins),
        "origin_dates": [dates[origin] for origin in origins],
        "context_end_dates": [dates[origin - 1] for origin in origins],
        "target_dates": [dates[origin : origin + H] for origin in origins],
        "quantile_levels": list(quantile_levels),
        "elapsed_seconds": elapsed_seconds,
    }


def naive_rolling_forecast(
    full_df: pd.DataFrame,
    origins: list[int],
    H: int,
    max_context: int,
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Menjalankan baseline naive persistence pada daftar origin yang sama.

    Bentuk keluarannya sengaja identik dengan :func:`rolling_forecast` agar
    ketiga skema dapat diperlakukan seragam pada tahap evaluasi. Konteks juga
    dipotong dengan aturan yang sama, sehingga estimasi sigma baseline memakai
    informasi yang persis tersedia bagi Chronos-2 pada origin tersebut.

    Args:
        full_df: Deret penuh dengan index tanggal asli.
        origins: Daftar posisi origin — WAJIB sama dengan yang dipakai model.
        H: Panjang horizon.
        max_context: Batas panjang konteks.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dictionary dengan kunci yang sama seperti :func:`rolling_forecast`.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("forecasting")

    target_column = get_target_column(config)
    quantile_levels = config["model"]["quantile_levels"]

    values = full_df[target_column].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(full_df.index)

    n_windows = len(origins)
    y_true = np.empty((n_windows, H), dtype=float)
    y_pred = np.empty((n_windows, H), dtype=float)
    y_pred_quantiles = np.empty((n_windows, H, len(quantile_levels)), dtype=float)

    started_at = time.perf_counter()

    for window_index, origin in enumerate(origins):
        context = values[max(0, origin - max_context) : origin]
        y_true[window_index] = values[origin : origin + H]
        y_pred[window_index] = naive_persistence(context, H)
        y_pred_quantiles[window_index] = naive_persistence_quantiles(
            context, H, quantile_levels, config=config
        )

    elapsed_seconds = time.perf_counter() - started_at
    logger.info(
        "Baseline naive persistence: %d jendela selesai dalam %.2f detik.",
        n_windows,
        elapsed_seconds,
    )

    return {
        "y_true": y_true,
        "y_pred_quantiles": y_pred_quantiles,
        "y_pred": y_pred,
        "origins": list(origins),
        "origin_dates": [dates[origin] for origin in origins],
        "context_end_dates": [dates[origin - 1] for origin in origins],
        "target_dates": [dates[origin : origin + H] for origin in origins],
        "quantile_levels": list(quantile_levels),
        "elapsed_seconds": elapsed_seconds,
    }


# =============================================================================
# Tuning hyperparameter fine-tuning (memakai data VALIDASI saja)
# =============================================================================


def weighted_quantile_loss(
    y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantile_levels: list[float]
) -> float:
    """Menghitung WQL (*weighted quantile loss*) seperti definisi AutoGluon.

    AutoGluon mendefinisikan::

        WQL = sum_t mean_k QL_{tau_k}(y_t, q_t) / sum_t |y_t|

    Sementara aproksimasi CRPS pada CLAUDE.md adalah ``2 * mean_k QL``, dan
    CRPS ternormalisasi membaginya dengan ``mean|y|``. Karena penyebut keduanya
    sebanding, berlaku hubungan eksak::

        WQL = normalized_crps / 2

    Fungsi ini karenanya memakai ulang :func:`src.evaluation.normalized_crps`
    daripada menghitung ulang pinball loss — satu implementasi, satu tempat
    untuk diuji.

    Args:
        y_true: Nilai aktual ``(n_windows, H)``.
        y_pred_quantiles: Ramalan kuantil ``(n_windows, H, n_quantiles)``.
        quantile_levels: Daftar level kuantil, menaik tegas.

    Returns:
        Nilai WQL (semakin kecil semakin baik).
    """
    return normalized_crps(y_true, y_pred_quantiles, quantile_levels) / 2.0


def tune_finetune_hyperparams(
    train_tsdf: TimeSeriesDataFrame,
    val_tsdf: TimeSeriesDataFrame,
    config: dict[str, Any] | None = None,
    horizon: int | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Memilih hyperparameter fine-tuning terbaik memakai DATA VALIDASI saja.

    **Inilah alasan keberadaan split validasi (keputusan D3).** Bobot model
    hanya boleh belajar dari data latih, tetapi keputusan "kombinasi
    hyperparameter mana yang dipakai" juga merupakan keputusan model. Bila
    keputusan itu diambil dari data uji, angka yang dilaporkan pada bab hasil
    bukan lagi estimasi performa pada data yang belum pernah dilihat, melainkan
    hasil terbaik yang sengaja dicari-cari di data uji — bentuk kebocoran yang
    paling sering luput. Karena itu:

    * setiap kandidat di-fine-tune HANYA pada ``train_tsdf``;
    * setiap kandidat dinilai HANYA pada periode ``val_tsdf``, memakai protokol
      rolling origin yang sama dengan evaluasi akhir (keputusan D4);
    * data uji tidak dibaca sama sekali oleh fungsi ini.

    Metrik pemilihan adalah ``finetune.selection_metric`` pada config (WQL,
    lihat :func:`weighted_quantile_loss`). Seluruh tabel hasil — termasuk
    kandidat yang kalah dan yang gagal — disimpan ke
    ``results/metrics/finetune_tuning.json`` agar proses pemilihannya dapat
    ditelusuri ulang.

    Catatan biaya: jumlah kandidat adalah hasil kali seluruh nilai pada
    ``finetune.grid``, dan tiap kandidat memerlukan satu fine-tuning penuh
    ditambah satu putaran rolling origin. Jendela validasi karenanya dijarangkan
    lewat ``finetune.val_stride`` dan ``finetune.max_val_windows``. Penjarangan
    ini hanya memengaruhi pemilihan hyperparameter, bukan evaluasi akhir.

    Args:
        train_tsdf: Data latih (``TimeSeriesDataFrame``) untuk fine-tuning.
        val_tsdf: Data validasi (``TimeSeriesDataFrame``); dipakai untuk
            menentukan panjang periode penilaian, bukan untuk melatih bobot.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        horizon: Panjang horizon H. Bila ``None``, diambil dari config.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dictionary berisi ``best_params``, ``best_score``, ``results`` (tabel
        seluruh kandidat), dan ``protocol``.

    Raises:
        ForecastingError: Bila grid kosong, memuat ``fine_tune_mode`` yang tidak
            didukung, atau seluruh kandidat gagal dijalankan.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("forecasting")
    if horizon is None:
        horizon = config["model"]["prediction_length"]

    finetune_config = config["finetune"]
    grid = finetune_config["grid"]

    learning_rates = list(grid["fine_tune_lr"])
    step_counts = list(grid["fine_tune_steps"])
    modes = list(grid["fine_tune_mode"])

    if not (learning_rates and step_counts and modes):
        raise ForecastingError("finetune.grid memuat daftar kosong.")

    invalid_modes = [mode for mode in modes if mode not in VALID_FINE_TUNE_MODES]
    if invalid_modes:
        raise ForecastingError(
            f"fine_tune_mode berikut tidak didukung chronos 2.3.1: {invalid_modes}. "
            f"Nilai yang valid: {list(VALID_FINE_TUNE_MODES)}."
        )

    # --- Deret gabungan latih+validasi: konteks jendela validasi harus dapat
    #     memakai riwayat data latih, persis seperti pada evaluasi akhir.
    #     Index gabungan ini memakai timestamp SINTETIS (lihat catatan indeks
    #     waktu pada docstring modul) karena tanggal asli tidak terbawa di dalam
    #     TimeSeriesDataFrame. Hal ini tidak berpengaruh: tahap tuning hanya
    #     memakai nilai ramalan dan aktual, tidak pernah tanggalnya. ---
    train_frame = train_tsdf.to_data_frame().reset_index(level=0, drop=True)
    val_frame = val_tsdf.to_data_frame().reset_index(level=0, drop=True)
    combined = pd.concat([train_frame, val_frame])
    combined.index.name = DATE_INDEX_NAME

    n_train = len(train_frame)
    origins = build_origins(
        len(combined), n_train, horizon, finetune_config["val_stride"]
    )
    max_val_windows = finetune_config["max_val_windows"]
    if max_val_windows is not None and len(origins) > max_val_windows:
        # Ambil merata sepanjang periode validasi, bukan hanya bagian awal
        keep = np.linspace(0, len(origins) - 1, max_val_windows).round().astype(int)
        origins = [origins[i] for i in sorted(set(keep.tolist()))]

    quantile_levels = config["model"]["quantile_levels"]
    max_context = config["model"]["max_context"]
    models_dir = resolve_path(config["paths"]["models_dir"]) / "finetune_tuning"

    combinations = list(itertools.product(learning_rates, step_counts, modes))

    logger.info("=" * 78)
    logger.info(
        "TUNING FINE-TUNING — %d kandidat, dinilai pada %d jendela VALIDASI "
        "(H=%d, stride=%d)",
        len(combinations),
        len(origins),
        horizon,
        finetune_config["val_stride"],
    )
    logger.info("Data uji TIDAK dibaca sama sekali pada tahap ini (aturan no. 2).")
    logger.info("=" * 78)

    results: list[dict[str, Any]] = []
    tuning_started_at = time.perf_counter()

    for candidate_index, (learning_rate, steps, mode) in enumerate(combinations):
        params = {
            "fine_tune_lr": float(learning_rate),
            "fine_tune_steps": int(steps),
            "fine_tune_mode": str(mode),
        }
        label = f"lr={learning_rate:g}_steps={steps}_mode={mode}"
        logger.info(
            "[%d/%d] Kandidat %s", candidate_index + 1, len(combinations), label
        )

        entry: dict[str, Any] = {
            "candidate_index": candidate_index,
            "label": label,
            **params,
        }

        try:
            candidate_started_at = time.perf_counter()
            predictor = build_finetuned_predictor(
                train_tsdf,
                params,
                config=config,
                horizon=horizon,
                path=str(models_dir / f"candidate_{candidate_index:02d}"),
                logger=logger,
            )
            fit_seconds = time.perf_counter() - candidate_started_at

            forecast = rolling_forecast(
                predictor,
                combined,
                n_train,
                horizon,
                finetune_config["val_stride"],
                max_context,
                config=config,
                origins=origins,
                logger=logger,
            )

            score = weighted_quantile_loss(
                forecast["y_true"], forecast["y_pred_quantiles"], quantile_levels
            )
            entry.update(
                {
                    "status": "ok",
                    "val_wql": float(score),
                    "val_mae": float(
                        np.abs(forecast["y_true"] - forecast["y_pred"]).mean()
                    ),
                    "fit_seconds": float(fit_seconds),
                    "forecast_seconds": float(forecast["elapsed_seconds"]),
                }
            )
            logger.info(
                "    WQL validasi = %.6f | MAE validasi = %.4f "
                "(fit %.1f s, ramal %.1f s)",
                entry["val_wql"],
                entry["val_mae"],
                fit_seconds,
                forecast["elapsed_seconds"],
            )
        except Exception as error:  # noqa: BLE001 — kandidat gagal tidak boleh menghentikan grid
            entry.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            logger.warning("    Kandidat GAGAL: %s", entry["error"])

        results.append(entry)

    successful = [entry for entry in results if entry["status"] == "ok"]
    if not successful:
        raise ForecastingError(
            "Seluruh kandidat fine-tuning gagal; tidak ada hyperparameter yang "
            "dapat dipilih. Periksa log untuk pesan galat tiap kandidat."
        )

    best = min(successful, key=lambda entry: entry["val_wql"])
    best_params = {
        "fine_tune_lr": best["fine_tune_lr"],
        "fine_tune_steps": best["fine_tune_steps"],
        "fine_tune_mode": best["fine_tune_mode"],
    }

    # Urutkan tabel dari terbaik ke terburuk agar mudah dibaca di laporan
    ranked = sorted(successful, key=lambda entry: entry["val_wql"])
    for rank, entry in enumerate(ranked, start=1):
        entry["rank"] = rank

    output: dict[str, Any] = {
        "selection_metric": finetune_config["selection_metric"],
        "selection_split": "val",
        "horizon": int(horizon),
        "best_params": best_params,
        "best_score": float(best["val_wql"]),
        "n_candidates": len(combinations),
        "n_failed": len(results) - len(successful),
        "total_seconds": float(time.perf_counter() - tuning_started_at),
        "protocol": {
            "grid": {
                "fine_tune_lr": learning_rates,
                "fine_tune_steps": step_counts,
                "fine_tune_mode": modes,
            },
            "val_stride": finetune_config["val_stride"],
            "max_val_windows": max_val_windows,
            "n_val_windows": len(origins),
            "origin_first": int(origins[0]),
            "origin_last": int(origins[-1]),
            "max_context": max_context,
            "note": (
                "Bobot dilatih hanya pada data latih; hyperparameter dipilih "
                "hanya dari data validasi; data uji tidak disentuh."
            ),
        },
        "results": results,
    }

    saved_path = save_json(
        output,
        resolve_path(config["paths"]["metrics_dir"]) / "finetune_tuning.json",
    )

    logger.info("-" * 78)
    logger.info("Hyperparameter terbaik (WQL validasi = %.6f): %s", best["val_wql"], best_params)
    logger.info("Tabel tuning lengkap disimpan di %s", saved_path)
    logger.info("=" * 78)

    return output


# =============================================================================
# Penyimpanan dan pemuatan ramalan mentah
# =============================================================================


def forecast_file_path(
    scheme: str, horizon: int, split: str, config: dict[str, Any]
) -> Path:
    """Menentukan lokasi berkas parquet ramalan mentah sebuah skema.

    Args:
        scheme: Nama skema (``zeroshot``, ``finetuned``, ``naive``).
        horizon: Panjang horizon H.
        split: Bagian data yang dievaluasi (``test`` atau ``val``).
        config: Konfigurasi project.

    Returns:
        Path absolut berkas parquet. Untuk ``split='test'`` namanya mengikuti
        spesifikasi ``{scheme}_H{H}.parquet``; bagian lain diberi akhiran nama
        split agar tidak menimpa hasil resmi.
    """
    suffix = "" if split == "test" else f"_{split}"
    return (
        resolve_path(config["paths"]["forecasts_dir"])
        / f"{scheme}_H{horizon}{suffix}.parquet"
    )


def forecast_to_frame(forecast: dict[str, Any], scheme: str) -> pd.DataFrame:
    """Meratakan hasil rolling forecast menjadi dataframe format panjang.

    Satu baris = satu pasangan (jendela, langkah horizon). Format ini dipilih
    karena mudah dibaca ulang, mudah difilter per horizon, dan tidak kehilangan
    satu pun angka ramalan mentah.

    Args:
        forecast: Keluaran :func:`rolling_forecast` atau
            :func:`naive_rolling_forecast`.
        scheme: Nama skema yang dicatat pada kolom ``scheme``.

    Returns:
        Dataframe dengan kolom ``scheme``, ``window``, ``origin_index``,
        ``origin_date``, ``context_end_date``, ``h``, ``target_date``,
        ``y_true``, dan satu kolom per level kuantil (mis. ``q0.5``).
    """
    y_true = forecast["y_true"]
    y_pred_quantiles = forecast["y_pred_quantiles"]
    quantile_levels = forecast["quantile_levels"]
    n_windows, horizon = y_true.shape

    records: dict[str, Any] = {
        "scheme": np.repeat(scheme, n_windows * horizon),
        "window": np.repeat(np.arange(n_windows), horizon),
        "origin_index": np.repeat(np.asarray(forecast["origins"]), horizon),
        "origin_date": np.repeat(
            pd.DatetimeIndex(forecast["origin_dates"]).to_numpy(), horizon
        ),
        "context_end_date": np.repeat(
            pd.DatetimeIndex(forecast["context_end_dates"]).to_numpy(), horizon
        ),
        "h": np.tile(np.arange(1, horizon + 1), n_windows),
        "target_date": np.concatenate(
            [pd.DatetimeIndex(dates).to_numpy() for dates in forecast["target_dates"]]
        ),
        "y_true": y_true.reshape(-1),
    }

    for quantile_index, level in enumerate(quantile_levels):
        records[f"q{level:g}"] = y_pred_quantiles[:, :, quantile_index].reshape(-1)

    return pd.DataFrame(records)


def save_forecast(
    forecast: dict[str, Any],
    scheme: str,
    horizon: int,
    split: str,
    config: dict[str, Any],
    logger: logging.Logger,
) -> Path:
    """Menyimpan ramalan mentah sebuah skema ke berkas parquet.

    Tujuannya agar seluruh tahap evaluasi, visualisasi, dan uji signifikansi
    dapat diulang tanpa menjalankan model lagi — penting karena satu putaran
    rolling origin di CPU memakan waktu menit hingga jam.

    Args:
        forecast: Keluaran rolling forecast.
        scheme: Nama skema.
        horizon: Panjang horizon H.
        split: Bagian data yang dievaluasi.
        config: Konfigurasi project.
        logger: Logger yang dipakai.

    Returns:
        Path absolut berkas yang tersimpan.
    """
    path = forecast_file_path(scheme, horizon, split, config)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame = forecast_to_frame(forecast, scheme)
    frame.to_parquet(path, index=False)

    logger.info(
        "Ramalan mentah skema '%s' disimpan: %d baris -> %s", scheme, len(frame), path
    )

    return path


def load_forecast_arrays(
    scheme: str,
    horizon: int | None = None,
    split: str = "test",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Memuat kembali ramalan mentah dari parquet menjadi array evaluasi.

    Fungsi ini adalah pasangan :func:`save_forecast`: keluarannya siap langsung
    diteruskan ke :func:`src.evaluation.aggregate_metrics`.

    Args:
        scheme: Nama skema.
        horizon: Panjang horizon H. Bila ``None``, diambil dari config.
        split: Bagian data yang dievaluasi.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.

    Returns:
        Dictionary dengan kunci ``y_true``, ``y_pred_quantiles``, ``y_pred``,
        ``origins``, ``origin_dates``, dan ``quantile_levels``.

    Raises:
        FileNotFoundError: Bila berkas parquet belum ada.
    """
    if config is None:
        config = load_config()
    if horizon is None:
        horizon = config["model"]["prediction_length"]

    path = forecast_file_path(scheme, horizon, split, config)
    if not path.is_file():
        raise FileNotFoundError(
            f"Berkas ramalan '{path}' tidak ditemukan. "
            f"Jalankan 'python -m src.forecasting' terlebih dahulu."
        )

    frame = pd.read_parquet(path).sort_values(["window", "h"])
    quantile_levels = config["model"]["quantile_levels"]
    quantile_columns = [f"q{level:g}" for level in quantile_levels]

    n_windows = int(frame["window"].nunique())
    y_true = frame["y_true"].to_numpy(dtype=float).reshape(n_windows, horizon)
    y_pred_quantiles = frame[quantile_columns].to_numpy(dtype=float).reshape(
        n_windows, horizon, len(quantile_levels)
    )
    median_index = find_quantile_index(
        quantile_levels, config["evaluation"]["point_quantile"]
    )

    first_rows = frame.drop_duplicates("window")

    return {
        "y_true": y_true,
        "y_pred_quantiles": y_pred_quantiles,
        "y_pred": y_pred_quantiles[:, :, median_index],
        "origins": first_rows["origin_index"].tolist(),
        "origin_dates": pd.DatetimeIndex(first_rows["origin_date"]).tolist(),
        "quantile_levels": quantile_levels,
    }


# =============================================================================
# Menjalankan seluruh skema
# =============================================================================


def run_all_schemes(
    config: dict[str, Any] | None = None,
    schemes: tuple[str, ...] = ALL_SCHEMES,
    split: str = "test",
    horizon: int | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Menjalankan zero-shot, fine-tuned, dan baseline pada jendela yang sama.

    Daftar origin dibentuk sekali saja lewat :func:`build_origins` lalu dipakai
    ulang oleh setiap skema, sehingga perbandingan antar skema — termasuk uji
    Diebold-Mariano — berpijak pada himpunan jendela yang identik (keputusan D4).

    Seluruh ramalan mentah disimpan ke
    ``results/forecasts/{scheme}_H{H}.parquet`` agar tahap evaluasi dapat
    diulang tanpa menjalankan model lagi.

    Args:
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        schemes: Skema yang dijalankan. Berguna untuk uji asap: cukup
            ``("zeroshot",)``.
        split: Bagian data yang dievaluasi. ``"test"`` untuk pelaporan akhir
            (hanya boleh dijalankan sekali, lihat aturan no. 2), ``"val"``
            untuk uji asap yang tidak menyentuh data uji.
        horizon: Panjang horizon H. Bila ``None``, diambil dari config.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dictionary ``{"forecasts": {...}, "summary": {...}}``.

    Raises:
        ForecastingError: Bila ``split`` atau nama skema tidak dikenali.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger(
            "forecasting", log_dir=config.get("paths", {}).get("logs_dir", "logs")
        )
    if horizon is None:
        horizon = config["model"]["prediction_length"]

    if split not in ("val", "test"):
        raise ForecastingError(f"split harus 'val' atau 'test', diterima '{split}'.")

    unknown = [scheme for scheme in schemes if scheme not in ALL_SCHEMES]
    if unknown:
        raise ForecastingError(
            f"Skema tidak dikenali: {unknown}. Pilihan: {list(ALL_SCHEMES)}."
        )

    stride = config["evaluation"]["stride"]
    max_context = config["model"]["max_context"]
    quantile_levels = config["model"]["quantile_levels"]

    logger.info("=" * 78)
    logger.info(
        "PERAMALAN CHRONOS-2 — split=%s, H=%d, stride=%d, skema=%s",
        split,
        horizon,
        stride,
        list(schemes),
    )
    logger.info("=" * 78)

    # --- Data: hanya bagian yang benar-benar dibutuhkan yang dibaca ---
    needed = ("train", "val") if split == "val" else ("train", "val", "test")
    frames = load_processed_frames(needed, config=config, logger=logger)
    full_df = pd.concat([frames[name] for name in needed])
    full_df.index.name = DATE_INDEX_NAME

    if split == "val":
        eval_start_idx = len(frames["train"])
        logger.info("Data UJI tidak dibaca sama sekali (uji asap pada validasi).")
    else:
        eval_start_idx = len(frames["train"]) + len(frames["val"])
        logger.warning(
            "Data UJI dibaca. Sesuai aturan no. 2, tahap ini hanya boleh "
            "dijalankan satu kali untuk pelaporan akhir."
        )

    # --- Daftar origin bersama: dibentuk SEKALI untuk seluruh skema (D4) ---
    origins = build_origins(len(full_df), eval_start_idx, horizon, stride)
    dates = pd.DatetimeIndex(full_df.index)
    logger.info(
        "Jendela bersama: %d origin | jendela pertama konteks s.d. %s -> sasaran "
        "%s s.d. %s | jendela terakhir sasaran %s s.d. %s",
        len(origins),
        dates[origins[0] - 1].date(),
        dates[origins[0]].date(),
        dates[origins[0] + horizon - 1].date(),
        dates[origins[-1]].date(),
        dates[origins[-1] + horizon - 1].date(),
    )

    train_tsdf = to_model_tsdf(frames["train"], config)

    forecasts: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {
        "split": split,
        "horizon": int(horizon),
        "stride": int(stride),
        "max_context": int(max_context),
        "n_windows": len(origins),
        "origin_first_date": str(dates[origins[0]].date()),
        "origin_last_date": str(dates[origins[-1]].date()),
        "quantile_levels": list(quantile_levels),
        "schemes": {},
    }

    for scheme in schemes:
        logger.info("-" * 78)
        logger.info("SKEMA: %s", scheme)
        logger.info("-" * 78)

        if scheme == SCHEME_ZEROSHOT:
            predictor = build_zeroshot_predictor(
                config=config,
                train_tsdf=train_tsdf,
                horizon=horizon,
                logger=logger,
            )
            forecast = rolling_forecast(
                predictor,
                full_df,
                eval_start_idx,
                horizon,
                stride,
                max_context,
                config=config,
                origins=origins,
                logger=logger,
            )

        elif scheme == SCHEME_FINETUNED:
            if not config["finetune"]["enabled"]:
                logger.warning("finetune.enabled=False — skema fine-tuned dilewati.")
                continue
            val_tsdf = to_model_tsdf(
                frames["val"], config, end_position=len(frames["train"]) + len(frames["val"])
            )
            tuning = tune_finetune_hyperparams(
                train_tsdf, val_tsdf, config=config, horizon=horizon, logger=logger
            )
            predictor = build_finetuned_predictor(
                train_tsdf,
                tuning["best_params"],
                config=config,
                horizon=horizon,
                logger=logger,
            )
            forecast = rolling_forecast(
                predictor,
                full_df,
                eval_start_idx,
                horizon,
                stride,
                max_context,
                config=config,
                origins=origins,
                logger=logger,
            )
            summary["finetune_best_params"] = tuning["best_params"]
            summary["finetune_best_val_wql"] = tuning["best_score"]

        else:  # SCHEME_NAIVE
            forecast = naive_rolling_forecast(
                full_df, origins, horizon, max_context, config=config, logger=logger
            )

        assert forecast["origins"] == list(origins), (
            f"Skema '{scheme}' memakai daftar origin yang berbeda — "
            f"melanggar keputusan D4."
        )

        path = save_forecast(forecast, scheme, horizon, split, config, logger)
        forecasts[scheme] = forecast

        # Ringkasan cepat untuk log. Metrik lengkap dihitung terpisah oleh
        # src.evaluation dari berkas parquet yang baru saja disimpan.
        mae_value = float(np.abs(forecast["y_true"] - forecast["y_pred"]).mean())
        crps_value = crps_approx(
            forecast["y_true"], forecast["y_pred_quantiles"], quantile_levels
        )
        summary["schemes"][scheme] = {
            "mae": mae_value,
            "crps": crps_value,
            "wql": weighted_quantile_loss(
                forecast["y_true"], forecast["y_pred_quantiles"], quantile_levels
            ),
            "elapsed_seconds": float(forecast["elapsed_seconds"]),
            "seconds_per_window": float(forecast["elapsed_seconds"] / len(origins)),
            "forecast_file": str(path),
        }
        logger.info(
            "Ringkasan '%s': MAE = %.4f | CRPS = %.4f | %.1f detik (%.2f detik/jendela)",
            scheme,
            mae_value,
            crps_value,
            forecast["elapsed_seconds"],
            forecast["elapsed_seconds"] / len(origins),
        )

    suffix = "" if split == "test" else f"_{split}"
    summary_path = save_json(
        summary,
        resolve_path(config["paths"]["metrics_dir"])
        / f"forecast_run_summary_H{horizon}{suffix}.json",
    )
    logger.info("Ringkasan run disimpan di %s", summary_path)
    logger.info("=" * 78)

    return {"forecasts": forecasts, "summary": summary}


# =============================================================================
# Titik masuk mandiri
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Membaca argumen baris perintah untuk eksekusi mandiri.

    Args:
        argv: Daftar argumen. Bila ``None``, diambil dari ``sys.argv``.

    Returns:
        Namespace berisi ``schemes``, ``split``, dan ``horizon``.
    """
    parser = argparse.ArgumentParser(
        description="Peramalan Chronos-2 dengan protokol rolling origin (D4)."
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=list(ALL_SCHEMES),
        default=[SCHEME_ZEROSHOT],
        help="Skema yang dijalankan (bawaan: zeroshot saja, untuk uji asap).",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help=(
            "Bagian data yang dievaluasi. Bawaan 'val' agar eksekusi mandiri "
            "tidak menyentuh data uji (aturan no. 2)."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Panjang horizon H. Bawaan: model.prediction_length pada config.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Titik masuk eksekusi mandiri modul.

    Secara bawaan menjalankan uji asap: skema zero-shot pada periode VALIDASI
    dengan H dari config. Data uji tidak disentuh kecuali ``--split test``
    diberikan secara eksplisit.

    Args:
        argv: Daftar argumen baris perintah.

    Returns:
        Keluaran :func:`run_all_schemes`.
    """
    args = parse_args(argv)

    config = load_config()
    logger = setup_logger(
        "forecasting", log_dir=config.get("paths", {}).get("logs_dir", "logs")
    )

    return run_all_schemes(
        config=config,
        schemes=tuple(args.schemes),
        split=args.split,
        horizon=args.horizon,
        logger=logger,
    )


if __name__ == "__main__":
    main()
