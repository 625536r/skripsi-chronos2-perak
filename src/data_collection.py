"""Pengunduhan data harga mentah dari Yahoo Finance.

Modul ini hanya bertanggung jawab mengambil data mentah dan menyimpannya apa
adanya ke ``data/raw/`` (immutable). Tidak ada penyelarasan tanggal, forward
fill, maupun transformasi lain di sini -- semua itu tugas ``src/preprocessing.py``.

Seri yang diunduh:
    - silver : SI=F      (target)
    - gold   : GC=F      (kovariat 1)
    - dxy    : DX-Y.NYB  (kovariat 2, fallback DX=F)

Cara menjalankan mandiri:
    python -m src.data_collection
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from src.utils import load_config, resolve_path, save_json, setup_logger

# Nama kolom keluaran standar dari download_ticker
CLOSE_COLUMN = "close"

# Nama index tanggal pada seluruh seri
DATE_INDEX_NAME = "date"


class DownloadError(RuntimeError):
    """Kegagalan pengunduhan data setelah seluruh percobaan ulang habis."""


class DataValidationError(ValueError):
    """Data berhasil diunduh tetapi tidak lolos pemeriksaan kewajaran."""


def _extract_close_series(
    raw_frame: pd.DataFrame, ticker: str, price_field: str
) -> pd.DataFrame:
    """Mengambil kolom harga penutupan dari hasil mentah yfinance.

    Sejak yfinance 1.x, ``yf.download`` selalu mengembalikan kolom MultiIndex
    dua level (``Price`` x ``Ticker``) walau hanya satu ticker yang diminta.
    Fungsi ini menangani kedua bentuk (MultiIndex maupun kolom datar) agar
    modul tidak rapuh terhadap perubahan perilaku library.

    Args:
        raw_frame: Dataframe mentah hasil ``yf.download``.
        ticker: Kode ticker yang diminta, dipakai untuk memilih pada level Ticker.
        price_field: Nama kolom harga yang diambil (mis. ``"Close"``).

    Returns:
        Dataframe satu kolom bernama ``close`` dengan index tanggal terurut.

    Raises:
        DataValidationError: Bila kolom harga yang diminta tidak ditemukan.
    """
    if isinstance(raw_frame.columns, pd.MultiIndex):
        # Cari level yang memuat nama field harga (biasanya level 0 / "Price")
        level_candidates = [
            level
            for level in range(raw_frame.columns.nlevels)
            if price_field in raw_frame.columns.get_level_values(level)
        ]
        if not level_candidates:
            raise DataValidationError(
                f"[{ticker}] Kolom '{price_field}' tidak ada pada hasil unduhan. "
                f"Kolom tersedia: {list(raw_frame.columns)}"
            )
        selected = raw_frame.xs(price_field, axis=1, level=level_candidates[0])

        # Setelah xs, sisa kolom adalah level Ticker. Ambil yang cocok bila ada.
        if isinstance(selected, pd.DataFrame):
            if ticker in selected.columns:
                close_series = selected[ticker]
            elif selected.shape[1] == 1:
                close_series = selected.iloc[:, 0]
            else:
                raise DataValidationError(
                    f"[{ticker}] Hasil unduhan memuat banyak ticker, "
                    f"tidak dapat memilih satu: {list(selected.columns)}"
                )
        else:
            close_series = selected
    else:
        if price_field not in raw_frame.columns:
            raise DataValidationError(
                f"[{ticker}] Kolom '{price_field}' tidak ada pada hasil unduhan. "
                f"Kolom tersedia: {list(raw_frame.columns)}"
            )
        close_series = raw_frame[price_field]

    frame = close_series.to_frame(name=CLOSE_COLUMN)

    # Normalisasi index: datetime tanpa timezone, dinormalkan ke tengah malam,
    # terurut naik, dan tanpa tanggal duplikat.
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame.index.name = DATE_INDEX_NAME
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    # Buang tanggal yang harganya kosong (hari libur bursa yang terbawa)
    frame = frame.dropna(subset=[CLOSE_COLUMN])
    frame[CLOSE_COLUMN] = frame[CLOSE_COLUMN].astype("float64")

    return frame


def _download_once(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    auto_adjust: bool,
    price_field: str,
    timeout: int,
) -> pd.DataFrame:
    """Melakukan satu kali percobaan unduhan tanpa retry.

    Args:
        ticker: Kode ticker Yahoo Finance.
        start: Tanggal awal (inklusif), format ``YYYY-MM-DD``.
        end: Tanggal akhir (eksklusif), format ``YYYY-MM-DD``.
        interval: Frekuensi data, mis. ``"1d"``.
        auto_adjust: Penyesuaian harga otomatis oleh yfinance.
        price_field: Nama kolom harga yang diambil.
        timeout: Batas waktu permintaan HTTP dalam detik.

    Returns:
        Dataframe satu kolom ``close``.

    Raises:
        DownloadError: Bila yfinance mengembalikan hasil kosong.
    """
    raw_frame = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
        timeout=timeout,
    )

    if raw_frame is None or raw_frame.empty:
        raise DownloadError(f"[{ticker}] yfinance mengembalikan hasil kosong.")

    return _extract_close_series(raw_frame, ticker=ticker, price_field=price_field)


def download_ticker(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    fallback_ticker: str | None = None,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    backoff_factor: float = 2.0,
    timeout: int = 30,
    auto_adjust: bool = False,
    price_field: str = "Close",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Mengunduh satu seri harga penutupan dari Yahoo Finance.

    Percobaan diulang sampai ``retries`` kali dengan jeda yang digandakan
    (backoff eksponensial). Bila seluruh percobaan pada ``ticker`` utama gagal
    dan ``fallback_ticker`` tersedia, unduhan diulang memakai ticker cadangan
    dan peristiwa ini DICATAT di log sebagai peringatan.

    Ticker yang benar-benar dipakai disimpan pada ``df.attrs["ticker"]`` agar
    dapat ditelusuri di metadata.

    Args:
        ticker: Kode ticker utama.
        start: Tanggal awal (inklusif), format ``YYYY-MM-DD``.
        end: Tanggal akhir (eksklusif), format ``YYYY-MM-DD``.
        interval: Frekuensi data, mis. ``"1d"``.
        fallback_ticker: Ticker cadangan bila ticker utama gagal.
        retries: Jumlah percobaan per ticker.
        backoff_seconds: Jeda awal sebelum percobaan ulang.
        backoff_factor: Pengali jeda antar percobaan.
        timeout: Batas waktu permintaan HTTP dalam detik.
        auto_adjust: Penyesuaian harga otomatis (WAJIB False sesuai CLAUDE.md).
        price_field: Nama kolom harga yang diambil.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dataframe satu kolom ``close`` dengan index tanggal terurut naik.

    Raises:
        DownloadError: Bila ticker utama maupun cadangan gagal diunduh.
    """
    if logger is None:
        logger = setup_logger("data_collection")

    candidates: list[tuple[str, bool]] = [(ticker, False)]
    if fallback_ticker:
        candidates.append((fallback_ticker, True))

    last_error: Exception | None = None

    for candidate_ticker, is_fallback in candidates:
        if is_fallback:
            logger.warning(
                "Ticker utama '%s' gagal diunduh (%s). "
                "Beralih ke ticker cadangan '%s'.",
                ticker,
                last_error,
                candidate_ticker,
            )

        delay = backoff_seconds
        for attempt in range(1, retries + 1):
            try:
                logger.info(
                    "Mengunduh %s (%s s/d %s, interval=%s) percobaan %d/%d",
                    candidate_ticker,
                    start,
                    end,
                    interval,
                    attempt,
                    retries,
                )
                frame = _download_once(
                    ticker=candidate_ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    price_field=price_field,
                    timeout=timeout,
                )
            except Exception as error:  # noqa: BLE001 - semua kegagalan ingin di-retry
                last_error = error
                logger.warning(
                    "Percobaan %d/%d untuk '%s' gagal: %s",
                    attempt,
                    retries,
                    candidate_ticker,
                    error,
                )
                if attempt < retries:
                    logger.info("Menunggu %.1f detik sebelum mencoba lagi...", delay)
                    time.sleep(delay)
                    delay *= backoff_factor
                continue

            frame.attrs["ticker"] = candidate_ticker
            frame.attrs["requested_ticker"] = ticker
            frame.attrs["used_fallback"] = is_fallback
            logger.info(
                "Berhasil mengunduh '%s': %d baris (%s s/d %s)",
                candidate_ticker,
                len(frame),
                frame.index[0].date(),
                frame.index[-1].date(),
            )
            if is_fallback:
                logger.warning(
                    "CATATAN: seri ini memakai ticker cadangan '%s', bukan '%s'.",
                    candidate_ticker,
                    ticker,
                )
            return frame

    attempted = " dan ".join(f"'{name}'" for name, _ in candidates)
    raise DownloadError(
        f"Gagal mengunduh {attempted} setelah {retries} percobaan per ticker. "
        f"Kesalahan terakhir: {last_error}"
    )


def validate_series(
    name: str,
    frame: pd.DataFrame,
    min_rows: int,
    max_rows: int,
    min_price: float,
    column: str = CLOSE_COLUMN,
) -> None:
    """Memeriksa kewajaran seri hasil unduhan.

    Pemeriksaan yang dilakukan:
        1. Seri tidak kosong.
        2. Jumlah baris berada dalam rentang wajar untuk 5 tahun data harian.
        3. Tidak ada nilai kosong.
        4. Tidak ada harga <= ``min_price``.

    Args:
        name: Nama seri untuk pesan kesalahan (mis. ``"silver"``).
        frame: Dataframe yang akan diperiksa.
        min_rows: Batas bawah jumlah baris yang dianggap wajar.
        max_rows: Batas atas jumlah baris yang dianggap wajar.
        min_price: Harga wajib lebih besar dari nilai ini.
        column: Nama kolom harga yang diperiksa.

    Raises:
        DataValidationError: Bila salah satu pemeriksaan gagal.
    """
    ticker = frame.attrs.get("ticker", "?")
    prefix = f"Validasi seri '{name}' (ticker '{ticker}') gagal:"

    if frame.empty:
        raise DataValidationError(f"{prefix} seri kosong, tidak ada satu pun baris.")

    row_count = len(frame)
    if not (min_rows <= row_count <= max_rows):
        raise DataValidationError(
            f"{prefix} jumlah baris {row_count} di luar rentang wajar "
            f"[{min_rows}, {max_rows}] untuk 5 tahun data harian. "
            f"Rentang tanggal terunduh: {frame.index[0].date()} s/d "
            f"{frame.index[-1].date()}. Periksa tanggal pada config atau "
            f"ketersediaan data ticker tersebut."
        )

    missing_count = int(frame[column].isna().sum())
    if missing_count > 0:
        missing_dates = frame.index[frame[column].isna()][:5]
        raise DataValidationError(
            f"{prefix} terdapat {missing_count} nilai kosong pada kolom "
            f"'{column}'. Contoh tanggal: "
            f"{[str(d.date()) for d in missing_dates]}"
        )

    invalid_mask = frame[column] <= min_price
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        invalid_dates = frame.index[invalid_mask][:5]
        raise DataValidationError(
            f"{prefix} terdapat {invalid_count} harga <= {min_price} pada kolom "
            f"'{column}'. Contoh tanggal: "
            f"{[str(d.date()) for d in invalid_dates]}"
        )


def _build_series_metadata(
    name: str,
    frame: pd.DataFrame,
    spec: dict[str, Any],
    price_column: str,
    output_file: str,
) -> dict[str, Any]:
    """Menyusun metadata satu seri hasil unduhan.

    Args:
        name: Nama seri (mis. ``"silver"``).
        frame: Dataframe seri yang sudah tervalidasi.
        spec: Spesifikasi ticker dari config.
        price_column: Nama kolom harga pada dataframe saat ini.
        output_file: Nama berkas CSV tempat seri disimpan.

    Returns:
        Dictionary metadata seri.
    """
    prices = frame[price_column]
    return {
        "name": name,
        "role": spec.get("role"),
        "requested_ticker": spec.get("ticker"),
        "actual_ticker": frame.attrs.get("ticker"),
        "used_fallback": bool(frame.attrs.get("used_fallback", False)),
        # Nama kolom sebagaimana tersimpan di berkas CSV
        "column": spec["column"],
        "file": output_file,
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "n_rows": int(len(frame)),
        "date_start": str(frame.index[0].date()),
        "date_end": str(frame.index[-1].date()),
        "price_min": float(prices.min()),
        "price_max": float(prices.max()),
        "price_mean": float(prices.mean()),
    }


def download_all(
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame]:
    """Mengunduh seluruh seri pada config, memvalidasi, dan menyimpannya.

    Setiap seri disimpan ke ``data/raw/{name}_raw.csv`` dengan nama kolom sesuai
    config (mis. ``silver_close``), dan ringkasan seluruh unduhan ditulis ke
    ``data/raw/download_metadata.json``.

    Args:
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dictionary ``{nama_seri: DataFrame}``. Tiap dataframe memiliki satu
        kolom harga dengan nama sesuai config dan index tanggal.

    Raises:
        DownloadError: Bila ada seri yang gagal diunduh.
        DataValidationError: Bila ada seri yang tidak lolos validasi.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger(
            "data_collection", log_dir=config.get("paths", {}).get("logs_dir", "logs")
        )

    data_config = config["data"]
    download_config = data_config["download"]
    validation_config = data_config["validation"]

    start = data_config["start_date"]
    end = data_config["end_date"]
    interval = data_config["interval"]

    raw_dir = resolve_path(data_config["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(
        "Mulai pengunduhan data: %s s/d %s (interval=%s, auto_adjust=%s)",
        start,
        end,
        interval,
        data_config["auto_adjust"],
    )
    logger.info("=" * 70)

    series: dict[str, pd.DataFrame] = {}
    metadata_series: dict[str, Any] = {}

    for name, spec in data_config["tickers"].items():
        column = spec["column"]

        frame = download_ticker(
            ticker=spec["ticker"],
            start=start,
            end=end,
            interval=interval,
            fallback_ticker=spec.get("fallback_ticker"),
            retries=download_config["retries"],
            backoff_seconds=download_config["backoff_seconds"],
            backoff_factor=download_config["backoff_factor"],
            timeout=download_config["timeout"],
            auto_adjust=data_config["auto_adjust"],
            price_field=data_config["price_field"],
            logger=logger,
        )

        validate_series(
            name=name,
            frame=frame,
            min_rows=validation_config["min_rows"],
            max_rows=validation_config["max_rows"],
            min_price=validation_config["min_price"],
        )
        logger.info("Seri '%s' lolos validasi (%d baris).", name, len(frame))

        output_path = raw_dir / f"{name}_raw.csv"
        metadata_series[name] = _build_series_metadata(
            name, frame, spec, CLOSE_COLUMN, output_path.name
        )

        # Ganti nama kolom ke penamaan final sesuai config sebelum disimpan
        frame = frame.rename(columns={CLOSE_COLUMN: column})
        frame.to_csv(output_path, index=True)
        logger.info("Seri '%s' disimpan ke %s", name, output_path)

        series[name] = frame

    metadata = {
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "source": "Yahoo Finance (yfinance)",
        "yfinance_version": yf.__version__,
        "request": {
            "start_date": start,
            "end_date": end,
            "interval": interval,
            "auto_adjust": data_config["auto_adjust"],
            "price_field": data_config["price_field"],
        },
        "validation": dict(validation_config),
        "series": metadata_series,
    }

    metadata_path = save_json(
        metadata, data_config.get("metadata_file", "data/raw/download_metadata.json")
    )
    logger.info("Metadata unduhan disimpan ke %s", metadata_path)

    return series


def summarize_series(series: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Menyusun ringkasan seri: jumlah baris, rentang tanggal, dan irisan tanggal.

    Args:
        series: Dictionary ``{nama_seri: DataFrame}`` hasil ``download_all``.

    Returns:
        Dictionary ringkasan per seri dan ringkasan irisan tanggal antar seri.
    """
    per_series: dict[str, Any] = {}
    index_sets: dict[str, set[pd.Timestamp]] = {}

    for name, frame in series.items():
        index_sets[name] = set(frame.index)
        per_series[name] = {
            "n_rows": int(len(frame)),
            "date_start": str(frame.index[0].date()),
            "date_end": str(frame.index[-1].date()),
            "column": frame.columns[0],
        }

    all_indexes = list(index_sets.values())
    union_dates: set[pd.Timestamp] = set().union(*all_indexes)
    intersection_dates: set[pd.Timestamp] = set(all_indexes[0]).intersection(*all_indexes)
    non_overlapping = union_dates - intersection_dates

    # Rincian: tanggal yang dimiliki satu seri tapi tidak dimiliki seri lain
    missing_per_series = {
        name: int(len(union_dates - dates)) for name, dates in index_sets.items()
    }
    extra_per_series = {
        name: int(len(dates - intersection_dates)) for name, dates in index_sets.items()
    }

    return {
        "per_series": per_series,
        "overlap": {
            "n_union": int(len(union_dates)),
            "n_intersection": int(len(intersection_dates)),
            "n_non_overlapping": int(len(non_overlapping)),
            "missing_vs_union_per_series": missing_per_series,
            "extra_vs_intersection_per_series": extra_per_series,
        },
    }


def _print_summary(summary: dict[str, Any], logger: logging.Logger) -> None:
    """Mencetak ringkasan hasil unduhan ke log.

    Args:
        summary: Hasil ``summarize_series``.
        logger: Logger tujuan.
    """
    logger.info("=" * 70)
    logger.info("RINGKASAN HASIL UNDUHAN")
    logger.info("=" * 70)

    logger.info("%-8s %-16s %8s  %-12s %-12s", "SERI", "KOLOM", "BARIS", "MULAI", "AKHIR")
    for name, info in summary["per_series"].items():
        logger.info(
            "%-8s %-16s %8d  %-12s %-12s",
            name,
            info["column"],
            info["n_rows"],
            info["date_start"],
            info["date_end"],
        )

    overlap = summary["overlap"]
    logger.info("-" * 70)
    logger.info("Gabungan tanggal (union)      : %d", overlap["n_union"])
    logger.info("Irisan tanggal (ketiga seri)  : %d", overlap["n_intersection"])
    logger.info(
        "Tanggal TIDAK beririsan       : %d  (union - irisan)",
        overlap["n_non_overlapping"],
    )
    logger.info("Rincian tanggal yang hilang dibanding union:")
    for name, count in overlap["missing_vs_union_per_series"].items():
        logger.info("  %-8s kekurangan %4d tanggal", name, count)
    logger.info("=" * 70)


def main() -> dict[str, pd.DataFrame]:
    """Titik masuk untuk eksekusi mandiri modul.

    Returns:
        Dictionary ``{nama_seri: DataFrame}`` hasil unduhan.
    """
    config = load_config()
    logger = setup_logger(
        "data_collection", log_dir=config.get("paths", {}).get("logs_dir", "logs")
    )

    series = download_all(config=config, logger=logger)
    summary = summarize_series(series)
    _print_summary(summary, logger)

    return series


if __name__ == "__main__":
    main()
