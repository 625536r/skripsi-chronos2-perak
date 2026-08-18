"""Penyelarasan tanggal, validasi kualitas data, dan pembagian data kronologis.

Modul ini menerapkan dua keputusan metodologis pada CLAUDE.md:

D1 -- Penyelarasan tanggal
    Outer join pada indeks tanggal -> forward fill -> buang baris yang nilai
    targetnya (``silver_close``) merupakan hasil forward fill. Alasannya, bila
    target ikut di-forward-fill akan muncul segmen datar buatan yang secara
    artifisial menaikkan akurasi model. Kovariat boleh di-forward-fill karena
    itu adalah informasi terakhir yang tersedia dan tidak melihat masa depan.

D3 -- Pembagian data
    Kronologis tanpa acak: 70% latih / 15% validasi / 15% uji.

Cara menjalankan mandiri:
    python -m src.preprocessing
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame

from src.data_collection import DATE_INDEX_NAME
from src.utils import load_config, resolve_path, save_json, setup_logger


class PreprocessingError(ValueError):
    """Kegagalan pada tahap penyelarasan atau pembagian data."""


def get_target_column(config: dict[str, Any]) -> str:
    """Mengambil nama kolom target dari config berdasarkan peran ``target``.

    Args:
        config: Konfigurasi project.

    Returns:
        Nama kolom target (mis. ``"silver_close"``).

    Raises:
        PreprocessingError: Bila tidak ada tepat satu seri berperan target.
    """
    targets = [
        spec["column"]
        for spec in config["data"]["tickers"].values()
        if spec.get("role") == "target"
    ]
    if len(targets) != 1:
        raise PreprocessingError(
            f"Config harus memuat tepat satu seri dengan role='target', "
            f"ditemukan {len(targets)}: {targets}"
        )
    return targets[0]


def get_covariate_columns(config: dict[str, Any]) -> list[str]:
    """Mengambil daftar nama kolom kovariat dari config.

    Args:
        config: Konfigurasi project.

    Returns:
        Daftar nama kolom kovariat sesuai urutan pada config.
    """
    return [
        spec["column"]
        for spec in config["data"]["tickers"].values()
        if spec.get("role") == "covariate"
    ]


def load_raw_series(
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame]:
    """Memuat kembali seri mentah dari ``data/raw/{name}_raw.csv``.

    Args:
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dictionary ``{nama_seri: DataFrame}`` dengan index tanggal.

    Raises:
        FileNotFoundError: Bila berkas mentah belum ada.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("preprocessing")

    raw_dir = resolve_path(config["data"]["raw_dir"])
    series: dict[str, pd.DataFrame] = {}

    for name in config["data"]["tickers"]:
        path = raw_dir / f"{name}_raw.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Berkas mentah '{path}' tidak ditemukan. "
                f"Jalankan 'python -m src.data_collection' terlebih dahulu."
            )
        frame = pd.read_csv(
            path, index_col=DATE_INDEX_NAME, parse_dates=[DATE_INDEX_NAME]
        )
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        frame.index.name = DATE_INDEX_NAME
        series[name] = frame.sort_index()
        logger.info("Memuat seri mentah '%s': %d baris dari %s", name, len(frame), path)

    return series


def align_series(
    dfs: dict[str, pd.DataFrame],
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Menyelaraskan seluruh seri pada satu indeks tanggal sesuai keputusan D1.

    Urutan langkah:
        1. Outer join seluruh seri pada indeks tanggal (union tanggal).
        2. Catat sel mana yang kosong SEBELUM forward fill.
        3. Forward fill seluruh kolom harga.
        4. Bentuk kolom flag ``<kovariat>_ffilled`` bernilai True bila nilainya
           benar-benar berasal dari forward fill.
        5. Buang baris yang nilai targetnya hasil forward fill atau tetap NaN.
        6. Buang baris awal yang masih memuat NaN (sebelum observasi pertama
           kovariat tersedia).

    Statistik tiap tahap disimpan pada ``df.attrs["alignment_stats"]`` agar dapat
    dilaporkan oleh :func:`validate_data`.

    Args:
        dfs: Dictionary ``{nama_seri: DataFrame}`` hasil pengunduhan.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dataframe tergabung dengan kolom harga dan kolom flag forward fill.

    Raises:
        PreprocessingError: Bila hasil penyelarasan kosong atau kolom target
            tidak ditemukan.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("preprocessing")

    preprocessing_config = config["preprocessing"]
    target_column = get_target_column(config)
    flag_columns: dict[str, str] = preprocessing_config["ffill_flag_columns"]

    # --- Langkah 1: outer join pada indeks tanggal ---
    joined = pd.concat(
        list(dfs.values()), axis=1, join=preprocessing_config["join_how"]
    ).sort_index()
    joined.index.name = DATE_INDEX_NAME

    if joined.empty:
        raise PreprocessingError("Hasil outer join kosong, tidak ada tanggal sama sekali.")
    if target_column not in joined.columns:
        raise PreprocessingError(
            f"Kolom target '{target_column}' tidak ada pada hasil join. "
            f"Kolom tersedia: {list(joined.columns)}"
        )

    n_union = len(joined)
    logger.info("Outer join menghasilkan %d tanggal (union).", n_union)

    # --- Langkah 2: catat sel kosong SEBELUM forward fill ---
    missing_before = joined.isna()

    # --- Langkah 3: forward fill seluruh kolom harga ---
    if preprocessing_config["fill_method"] != "ffill":
        raise PreprocessingError(
            f"fill_method '{preprocessing_config['fill_method']}' tidak didukung; "
            f"keputusan D1 mensyaratkan 'ffill'."
        )
    filled = joined.ffill()

    # --- Langkah 4: kolom flag forward fill untuk kovariat ---
    # Flag bernilai True hanya bila nilai semula kosong DAN berhasil diisi ffill.
    ffill_counts_before_drop: dict[str, int] = {}
    for series_name, flag_column in flag_columns.items():
        price_column = config["data"]["tickers"][series_name]["column"]
        flag = missing_before[price_column] & filled[price_column].notna()
        filled[flag_column] = flag
        ffill_counts_before_drop[flag_column] = int(flag.sum())

    # --- Langkah 5: buang baris yang targetnya hasil ffill atau tetap NaN ---
    target_was_missing = missing_before[target_column]
    n_dropped_target_ffilled = int(target_was_missing.sum())

    if preprocessing_config["drop_ffilled_target"]:
        filled = filled.loc[~target_was_missing]
    else:
        logger.warning(
            "drop_ffilled_target=False -- menyimpang dari keputusan D1, "
            "hasil evaluasi berpotensi bias optimistis."
        )
        n_dropped_target_ffilled = 0

    n_after_target_drop = len(filled)

    # --- Langkah 6: buang baris awal yang masih memuat NaN ---
    price_columns = [target_column, *get_covariate_columns(config)]
    rows_with_nan = filled[price_columns].isna().any(axis=1)
    n_dropped_leading_nan = int(rows_with_nan.sum())
    aligned = filled.loc[~rows_with_nan].copy()

    if aligned.empty:
        raise PreprocessingError(
            "Seluruh baris terbuang setelah penyelarasan. "
            "Periksa apakah rentang tanggal ketiga seri benar-benar beririsan."
        )

    # Susun ulang urutan kolom: harga dulu, lalu flag
    aligned = aligned[price_columns + list(flag_columns.values())]

    # Hitungan forward fill yang benar-benar bertahan di data akhir. Ini bisa
    # lebih kecil daripada hitungan sebelum pembuangan, karena sebuah tanggal
    # yang kovariatnya di-ffill bisa saja ikut terbuang oleh aturan D1 (target
    # pada tanggal itu ternyata juga hasil forward fill).
    ffill_counts_final = {
        flag_column: int(aligned[flag_column].sum())
        for flag_column in flag_columns.values()
    }

    aligned.attrs["alignment_stats"] = {
        "n_union_dates": n_union,
        "n_dropped_target_ffilled": n_dropped_target_ffilled,
        "n_after_target_drop": n_after_target_drop,
        "n_dropped_leading_nan": n_dropped_leading_nan,
        "n_final": int(len(aligned)),
        "n_total_dropped": int(n_union - len(aligned)),
        "ffill_counts": ffill_counts_final,
        "ffill_counts_before_drop": ffill_counts_before_drop,
        "target_column": target_column,
        "covariate_columns": get_covariate_columns(config),
        "flag_columns": list(flag_columns.values()),
    }

    logger.info(
        "D1: %d baris dibuang karena target hasil forward fill.",
        n_dropped_target_ffilled,
    )
    logger.info(
        "D1: %d baris dibuang karena masih memuat NaN di awal periode.",
        n_dropped_leading_nan,
    )
    for flag_column, count in ffill_counts_final.items():
        logger.info(
            "Kovariat '%s' hasil forward fill: %d baris bertahan "
            "(%d sebelum pembuangan D1).",
            flag_column,
            count,
            ffill_counts_before_drop[flag_column],
        )
    logger.info("Hasil akhir penyelarasan: %d baris.", len(aligned))

    return aligned


def validate_data(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Menyusun laporan kualitas data hasil penyelarasan.

    Pemeriksaan yang dilakukan:
        - tanggal duplikat
        - tanggal tidak berurutan (index tidak monoton naik)
        - harga <= 0 per kolom
        - NaN tersisa per kolom
        - gap kalender terpanjang antar tanggal berurutan
        - jumlah baris yang dibuang pada tiap tahap penyelarasan

    Args:
        df: Dataframe hasil :func:`align_series`.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.

    Returns:
        Dictionary laporan. Kunci ``passed`` bernilai False bila ada masalah
        yang terdeteksi, dan ``issues`` memuat rangkuman masalahnya.
    """
    if config is None:
        config = load_config()

    target_column = get_target_column(config)
    price_columns = [target_column, *get_covariate_columns(config)]
    min_price = config["data"]["validation"]["min_price"]

    index = pd.DatetimeIndex(df.index)

    # --- Tanggal duplikat ---
    duplicated_mask = index.duplicated(keep=False)
    duplicate_dates = sorted({str(d.date()) for d in index[duplicated_mask]})

    # --- Tanggal tidak berurutan ---
    is_monotonic = bool(index.is_monotonic_increasing)
    unordered_positions = [
        int(i) for i in range(1, len(index)) if index[i] <= index[i - 1]
    ]

    # --- Harga <= 0 dan NaN tersisa ---
    non_positive: dict[str, int] = {}
    remaining_nan: dict[str, int] = {}
    for column in price_columns:
        non_positive[column] = int((df[column] <= min_price).sum())
        remaining_nan[column] = int(df[column].isna().sum())

    # --- Gap kalender terpanjang ---
    gap_report: dict[str, Any] = {}
    if len(index) > 1:
        gaps = index.to_series().diff().dt.days.dropna()
        max_gap = int(gaps.max())
        max_gap_position = gaps.idxmax()
        position = index.get_loc(max_gap_position)
        gap_report = {
            "max_gap_days": max_gap,
            "gap_from": str(index[position - 1].date()),
            "gap_to": str(index[position].date()),
            "median_gap_days": float(gaps.median()),
            # Distribusi gap: mayoritas 1 hari (hari kerja), 3 hari (akhir pekan)
            "gap_day_counts": {
                str(int(k)): int(v) for k, v in gaps.value_counts().sort_index().items()
            },
        }

    stats = df.attrs.get("alignment_stats", {})

    issues: list[str] = []
    if duplicate_dates:
        issues.append(f"terdapat {len(duplicate_dates)} tanggal duplikat")
    if not is_monotonic:
        issues.append(f"terdapat {len(unordered_positions)} tanggal tidak berurutan")
    total_non_positive = sum(non_positive.values())
    if total_non_positive:
        issues.append(f"terdapat {total_non_positive} harga <= {min_price}")
    total_nan = sum(remaining_nan.values())
    if total_nan:
        issues.append(f"terdapat {total_nan} nilai NaN tersisa")

    return {
        "n_rows": int(len(df)),
        "date_start": str(index[0].date()) if len(index) else None,
        "date_end": str(index[-1].date()) if len(index) else None,
        "columns": list(df.columns),
        "duplicate_dates": {
            "count": len(duplicate_dates),
            "examples": duplicate_dates[:10],
        },
        "ordering": {
            "is_monotonic_increasing": is_monotonic,
            "n_unordered": len(unordered_positions),
        },
        "non_positive_prices": non_positive,
        "remaining_nan": remaining_nan,
        "calendar_gap": gap_report,
        "rows_dropped": {
            "n_union_dates": stats.get("n_union_dates"),
            "dropped_target_ffilled_D1": stats.get("n_dropped_target_ffilled"),
            "dropped_leading_nan": stats.get("n_dropped_leading_nan"),
            "n_final": stats.get("n_final"),
            "n_total_dropped": stats.get("n_total_dropped"),
        },
        "ffill_counts": stats.get("ffill_counts", {}),
        "ffill_counts_before_drop": stats.get("ffill_counts_before_drop", {}),
        "passed": not issues,
        "issues": issues,
    }


def chronological_split(
    df: pd.DataFrame,
    ratios: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Membagi data secara kronologis menjadi latih/validasi/uji (keputusan D3).

    Pembagian murni berurutan tanpa pengacakan. Sisa pembulatan dialokasikan ke
    bagian uji agar tidak ada baris yang hilang.

    Args:
        df: Dataframe hasil :func:`align_series`, sudah terurut menaik.
        ratios: Dictionary rasio dengan kunci ``train_ratio``, ``val_ratio``,
            ``test_ratio``. Bila ``None``, diambil dari config.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Tuple ``(train, val, test)``.

    Raises:
        PreprocessingError: Bila rasio tidak berjumlah 1, data terlalu sedikit,
            atau pengacakan diaktifkan pada config.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("preprocessing")
    if ratios is None:
        ratios = config["split"]

    if config["split"].get("shuffle", False):
        raise PreprocessingError(
            "split.shuffle=True melanggar keputusan D3 (pembagian wajib kronologis)."
        )

    train_ratio = float(ratios["train_ratio"])
    val_ratio = float(ratios["val_ratio"])
    test_ratio = float(ratios["test_ratio"])

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise PreprocessingError(
            f"Jumlah rasio split harus 1.0, diperoleh {total_ratio} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )

    if not df.index.is_monotonic_increasing:
        raise PreprocessingError(
            "Index tanggal tidak terurut menaik; pembagian kronologis tidak sah."
        )

    n_total = len(df)
    if n_total < 3:
        raise PreprocessingError(
            f"Data terlalu sedikit untuk dibagi tiga ({n_total} baris)."
        )

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # Sisa pembulatan masuk ke bagian uji
    n_test = n_total - n_train - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise PreprocessingError(
            f"Pembagian menghasilkan bagian kosong "
            f"(train={n_train}, val={n_val}, test={n_test}) dari {n_total} baris."
        )

    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()

    # --- Pemeriksaan: tidak boleh ada tanggal yang tumpang tindih antar split ---
    train_dates, val_dates, test_dates = set(train.index), set(val.index), set(test.index)
    assert not (train_dates & val_dates), "Tanggal latih dan validasi tumpang tindih."
    assert not (val_dates & test_dates), "Tanggal validasi dan uji tumpang tindih."
    assert not (train_dates & test_dates), "Tanggal latih dan uji tumpang tindih."

    # --- Pemeriksaan: urutan kronologis antar split benar-benar terjaga ---
    assert train.index[-1] < val.index[0], "Batas latih/validasi tidak kronologis."
    assert val.index[-1] < test.index[0], "Batas validasi/uji tidak kronologis."

    # --- Pemeriksaan: tidak ada baris yang hilang ---
    assert len(train) + len(val) + len(test) == n_total, "Jumlah baris split tidak utuh."

    logger.info(
        "Split D3 -> latih %d (%s s/d %s) | validasi %d (%s s/d %s) | uji %d (%s s/d %s)",
        len(train),
        train.index[0].date(),
        train.index[-1].date(),
        len(val),
        val.index[0].date(),
        val.index[-1].date(),
        len(test),
        test.index[0].date(),
        test.index[-1].date(),
    )

    return train, val, test


def to_timeseries_dataframe(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> TimeSeriesDataFrame:
    """Mengubah dataframe hasil penyelarasan menjadi ``TimeSeriesDataFrame``.

    Bentuk keluaran mengikuti API AutoGluon-TimeSeries 1.6.x:
    ``TimeSeriesDataFrame.from_data_frame(df, id_column=..., timestamp_column=...)``
    yang mensyaratkan dataframe datar dengan kolom ``item_id`` dan ``timestamp``.

    Penelitian ini hanya memodelkan satu deret, sehingga ``item_id`` tunggal
    (nilainya dari ``model.item_id`` pada config).

    Kolom yang disertakan hanya kolom harga: target ``silver_close`` serta
    kovariat ``gold_close`` dan ``dxy_close``. Kolom flag ``*_ffilled`` sengaja
    TIDAK disertakan karena sifatnya metadata pelaporan, bukan variabel model.

    Catatan penting terkait keputusan D2: penentuan mana kolom target dan mana
    kovariat TIDAK dilakukan di sini, melainkan saat pembuatan
    ``TimeSeriesPredictor(target=..., known_covariates_names=[])``. Seluruh kolom
    selain target otomatis diperlakukan sebagai past-only covariates -- persis
    yang dikehendaki.

    Args:
        df: Dataframe dengan index tanggal (hasil align/split).
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Objek ``TimeSeriesDataFrame`` berisi satu item.

    Raises:
        PreprocessingError: Bila ada kolom harga yang tidak ditemukan.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("preprocessing")

    target_column = get_target_column(config)
    covariate_columns = get_covariate_columns(config)
    price_columns = [target_column, *covariate_columns]

    missing = [column for column in price_columns if column not in df.columns]
    if missing:
        raise PreprocessingError(
            f"Kolom berikut tidak ada pada dataframe: {missing}. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    flat = df[price_columns].reset_index()
    flat = flat.rename(columns={DATE_INDEX_NAME: TimeSeriesDataFrame.TIMESTAMP})
    flat[TimeSeriesDataFrame.ITEMID] = config["model"]["item_id"]

    ts_df = TimeSeriesDataFrame.from_data_frame(
        flat,
        id_column=TimeSeriesDataFrame.ITEMID,
        timestamp_column=TimeSeriesDataFrame.TIMESTAMP,
    )

    logger.info(
        "TimeSeriesDataFrame dibentuk: %d item, %d baris, kolom %s, freq=%r",
        ts_df.num_items,
        len(ts_df),
        list(ts_df.columns),
        ts_df.freq,
    )

    return ts_df


def _save_splits(
    aligned: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, str]:
    """Menyimpan hasil penyelarasan dan pembagian data ke ``data/processed/``.

    Args:
        aligned: Dataframe hasil penyelarasan.
        train: Bagian latih.
        val: Bagian validasi.
        test: Bagian uji.
        config: Konfigurasi project.
        logger: Logger yang dipakai.

    Returns:
        Dictionary ``{nama: path}`` berkas yang tersimpan.
    """
    processed_dir = resolve_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    frames = {"aligned": aligned, "train": train, "val": val, "test": test}
    saved: dict[str, str] = {}

    for name, frame in frames.items():
        path = processed_dir / f"{name}.csv"
        frame.to_csv(path, index=True)
        saved[name] = str(path)
        logger.info("Menyimpan %-8s %5d baris -> %s", name, len(frame), path)

    return saved


def _build_split_report(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    """Menyusun ringkasan pembagian data untuk laporan kualitas.

    Args:
        train: Bagian latih.
        val: Bagian validasi.
        test: Bagian uji.
        config: Konfigurasi project.

    Returns:
        Dictionary ringkasan tiap bagian beserta rasio aktualnya.
    """
    n_total = len(train) + len(val) + len(test)
    report: dict[str, Any] = {
        "n_total": n_total,
        "ratios_config": dict(config["split"]),
        "splits": {},
    }

    for name, frame in (("train", train), ("val", val), ("test", test)):
        report["splits"][name] = {
            "n_rows": int(len(frame)),
            "ratio_actual": round(len(frame) / n_total, 6),
            "date_start": str(frame.index[0].date()),
            "date_end": str(frame.index[-1].date()),
        }

    return report


def main() -> dict[str, Any]:
    """Titik masuk untuk eksekusi mandiri modul.

    Menjalankan seluruh tahap: memuat seri mentah, menyelaraskan (D1),
    memvalidasi, membagi kronologis (D3), menyimpan hasil, dan mencetak
    ringkasan.

    Returns:
        Dictionary berisi dataframe hasil dan laporan kualitas data.
    """
    config = load_config()
    logger = setup_logger(
        "preprocessing", log_dir=config.get("paths", {}).get("logs_dir", "logs")
    )

    logger.info("=" * 78)
    logger.info("PREPROCESSING: penyelarasan D1 + pembagian D3")
    logger.info("=" * 78)

    raw_series = load_raw_series(config=config, logger=logger)
    aligned = align_series(raw_series, config=config, logger=logger)

    quality_report = validate_data(aligned, config=config)
    if not quality_report["passed"]:
        logger.warning("Laporan kualitas menemukan masalah: %s", quality_report["issues"])
    else:
        logger.info("Laporan kualitas: seluruh pemeriksaan lolos.")

    train, val, test = chronological_split(aligned, config=config, logger=logger)

    saved_files = _save_splits(aligned, train, val, test, config, logger)

    # Verifikasi bahwa tiap bagian dapat dikonversi ke format AutoGluon
    ts_frames = {
        name: to_timeseries_dataframe(frame, config=config, logger=logger)
        for name, frame in (("train", train), ("val", val), ("test", test))
    }

    quality_report["split"] = _build_split_report(train, val, test, config)
    quality_report["files"] = saved_files
    quality_report["timeseries_dataframe"] = {
        name: {
            "num_items": int(ts.num_items),
            "n_rows": int(len(ts)),
            "columns": list(ts.columns),
            "freq": ts.freq,
        }
        for name, ts in ts_frames.items()
    }

    report_path = save_json(
        quality_report,
        resolve_path(config["paths"]["metrics_dir"]) / "data_quality_report.json",
    )
    logger.info("Laporan kualitas data disimpan ke %s", report_path)

    _print_summary(quality_report, logger)

    return {
        "aligned": aligned,
        "train": train,
        "val": val,
        "test": test,
        "report": quality_report,
    }


def _print_summary(report: dict[str, Any], logger: logging.Logger) -> None:
    """Mencetak ringkasan preprocessing ke log.

    Args:
        report: Laporan kualitas data lengkap.
        logger: Logger tujuan.
    """
    dropped = report["rows_dropped"]
    logger.info("=" * 78)
    logger.info("RINGKASAN PREPROCESSING")
    logger.info("=" * 78)

    logger.info("Baris dibuang karena aturan D1:")
    logger.info("  Union tanggal awal (outer join)      : %d", dropped["n_union_dates"])
    logger.info(
        "  Dibuang: target hasil forward fill   : %d", dropped["dropped_target_ffilled_D1"]
    )
    logger.info(
        "  Dibuang: NaN tersisa di awal periode : %d", dropped["dropped_leading_nan"]
    )
    logger.info("  Total dibuang                        : %d", dropped["n_total_dropped"])
    logger.info("  Baris akhir                          : %d", dropped["n_final"])

    logger.info("-" * 78)
    logger.info("Kovariat yang diisi forward fill (sesuai D1, kovariat boleh di-ffill):")
    before = report.get("ffill_counts_before_drop", {})
    for flag_column, count in report["ffill_counts"].items():
        logger.info(
            "  %-16s %d baris bertahan di data akhir (%d sebelum pembuangan D1)",
            flag_column,
            count,
            before.get(flag_column, count),
        )

    logger.info("-" * 78)
    logger.info(
        "%-8s %8s %10s  %-12s %-12s", "SPLIT", "BARIS", "RASIO", "MULAI", "AKHIR"
    )
    for name, info in report["split"]["splits"].items():
        logger.info(
            "%-8s %8d %9.2f%%  %-12s %-12s",
            name,
            info["n_rows"],
            info["ratio_actual"] * 100,
            info["date_start"],
            info["date_end"],
        )
    logger.info("%-8s %8d", "TOTAL", report["split"]["n_total"])

    logger.info("-" * 78)
    gap = report.get("calendar_gap", {})
    logger.info(
        "Gap kalender terpanjang: %s hari (%s -> %s)",
        gap.get("max_gap_days"),
        gap.get("gap_from"),
        gap.get("gap_to"),
    )
    logger.info("Pemeriksaan kualitas lolos: %s", report["passed"])
    if report["issues"]:
        for issue in report["issues"]:
            logger.warning("  MASALAH: %s", issue)
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
