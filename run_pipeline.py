"""Orkestrasi seluruh pipeline penelitian, tahap 1 sampai 6.

Tahapan:

1. Pengumpulan data      -- unduh SI=F, GC=F, DX-Y.NYB dari Yahoo Finance
2. Preprocessing & split -- penyelarasan tanggal, pembagian kronologis
3. Mutual information    -- MI level dan MI log-return dari data latih
4. Peramalan             -- zero-shot, fine-tuned, dan baseline pada jendela sama
5. Evaluasi              -- metrik lengkap, uji Diebold-Mariano, tabel BAB IV
6. Visualisasi           -- seluruh gambar laporan ke results/figures/

Berkas ini hanya **mengatur urutan**. Seluruh logika berada di ``src/`` supaya
tiap modul tetap dapat diuji dan dijalankan sendiri.

PERINGATAN KEBOCORAN DATA. Tahap 4 adalah satu-satunya tempat data uji dibaca,
dan hanya sekali, untuk pelaporan akhir. Pemilihan hyperparameter fine-tuning
terjadi di tahap 4 juga tetapi memakai data validasi saja (lihat
:func:`src.forecasting.tune_finetune_hyperparams`). Jangan pernah menjalankan
ulang tahap 5 dengan konfigurasi berbeda demi mencari angka yang lebih bagus —
itu mengubah data uji menjadi data validasi.

Cara menjalankan:
    python run_pipeline.py                            # seluruh tahap, seluruh skema
    python run_pipeline.py --skip-download            # pakai data mentah yang ada
    python run_pipeline.py --horizon 5                # horizon lain
    python run_pipeline.py --scheme zeroshot naive    # sebagian skema saja
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation import aggregate_metrics, diebold_mariano
from src.forecasting import (
    ALL_SCHEMES,
    SCHEME_FINETUNED,
    SCHEME_NAIVE,
    SCHEME_ZEROSHOT,
    forecast_file_path,
    load_forecast_arrays,
    load_processed_frames,
    run_all_schemes,
)
from src.utils import load_config, log_environment, resolve_path, save_json, setup_logger
from src.visualization import generate_all_figures

# Nama tampil skema pada tabel BAB IV
SCHEME_DISPLAY: dict[str, str] = {
    SCHEME_NAIVE: "Naive Persistence",
    SCHEME_ZEROSHOT: "Chronos-2 Zero-Shot",
    SCHEME_FINETUNED: "Chronos-2 Fine-Tuned",
}

# Urutan baris pada tabel perbandingan: baseline dulu, lalu kedua skema model
TABLE_ORDER: tuple[str, ...] = (SCHEME_NAIVE, SCHEME_ZEROSHOT, SCHEME_FINETUNED)

# Pasangan skema yang diuji signifikansi bedanya.
# Pasangan pertama adalah uji utama penelitian; dua sisanya pelengkap.
DM_PAIRS: tuple[tuple[str, str, bool], ...] = (
    (SCHEME_ZEROSHOT, SCHEME_FINETUNED, True),
    (SCHEME_ZEROSHOT, SCHEME_NAIVE, False),
    (SCHEME_FINETUNED, SCHEME_NAIVE, False),
)


class PipelineError(RuntimeError):
    """Kegagalan pada salah satu tahap pipeline."""


# =============================================================================
# Utilitas tahap
# =============================================================================


def _log_stage(logger: logging.Logger, number: int, title: str) -> float:
    """Mencetak penanda awal sebuah tahap ke log.

    Args:
        logger: Logger yang dipakai.
        number: Nomor tahap.
        title: Judul tahap.

    Returns:
        Waktu mulai (``time.perf_counter``) untuk dipakai menghitung durasi.
    """
    logger.info("")
    logger.info("=" * 78)
    logger.info("TAHAP %d — %s", number, title)
    logger.info("=" * 78)
    return time.perf_counter()


def _log_stage_done(
    logger: logging.Logger, number: int, title: str, started_at: float
) -> float:
    """Mencetak penanda akhir sebuah tahap beserta durasinya.

    Args:
        logger: Logger yang dipakai.
        number: Nomor tahap.
        title: Judul tahap.
        started_at: Nilai kembalian :func:`_log_stage`.

    Returns:
        Durasi tahap dalam detik.
    """
    elapsed = time.perf_counter() - started_at
    logger.info("TAHAP %d (%s) selesai dalam %.1f detik.", number, title, elapsed)
    return elapsed


# =============================================================================
# Tahap 1-3
# =============================================================================


def stage_data_collection(
    config: dict[str, Any], logger: logging.Logger, skip: bool
) -> dict[str, Any]:
    """Tahap 1 — mengunduh seluruh seri harga dari Yahoo Finance.

    Args:
        config: Konfigurasi project.
        logger: Logger yang dipakai.
        skip: Bila ``True``, pengunduhan dilewati dan berkas mentah yang sudah
            ada dipakai apa adanya.

    Returns:
        Dictionary status tahap.

    Raises:
        PipelineError: Bila pengunduhan dilewati padahal berkas mentah belum ada.
    """
    if skip:
        raw_dir = resolve_path(config["data"]["raw_dir"])
        missing = [
            name
            for name in config["data"]["tickers"]
            if not (raw_dir / f"{name}_raw.csv").is_file()
        ]
        if missing:
            raise PipelineError(
                f"--skip-download diberikan, tetapi berkas mentah berikut belum "
                f"ada: {missing}. Jalankan tanpa --skip-download terlebih dahulu."
            )
        logger.info("Pengunduhan dilewati (--skip-download); memakai data/raw/ yang ada.")
        return {"status": "skipped"}

    from src.data_collection import main as download_main

    series = download_main()
    return {
        "status": "ok",
        "series": {name: int(len(frame)) for name, frame in series.items()},
    }


def stage_preprocessing(config: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    """Tahap 2 — penyelarasan tanggal dan pembagian data kronologis.

    Args:
        config: Konfigurasi project.
        logger: Logger yang dipakai.

    Returns:
        Dictionary ringkasan pembagian data.
    """
    from src.preprocessing import main as preprocessing_main

    result = preprocessing_main()
    report = result["report"]

    logger.info(
        "Hasil: %d baris final | latih %d | validasi %d | uji %d",
        report["n_rows"],
        len(result["train"]),
        len(result["val"]),
        len(result["test"]),
    )

    return {
        "status": "ok",
        "n_rows": report["n_rows"],
        "split": report["split"]["splits"],
        "quality_passed": report["passed"],
        "quality_issues": report["issues"],
    }


def stage_mutual_information(
    config: dict[str, Any], logger: logging.Logger
) -> dict[str, Any]:
    """Tahap 3 — mutual information pada data latih saja.

    Args:
        config: Konfigurasi project.
        logger: Logger yang dipakai.

    Returns:
        Dictionary ringkasan MI kedua varian.
    """
    from src.mutual_information import main as mi_main

    result = mi_main()

    summary: dict[str, Any] = {"status": "ok", "variants": {}}
    for mode, entry in result.get("results", {}).items():
        summary["variants"][mode] = {
            covariate: round(float(pair["mi_mean"]), 6)
            for covariate, pair in entry["pairs"].items()
        }
        logger.info("MI %-12s: %s", mode, summary["variants"][mode])

    return summary


# =============================================================================
# Tahap 4 — peramalan
# =============================================================================


def stage_forecasting(
    config: dict[str, Any],
    logger: logging.Logger,
    schemes: tuple[str, ...],
    horizon: int,
) -> dict[str, Any]:
    """Tahap 4 — menjalankan seluruh skema pada periode uji.

    Inilah satu-satunya tahap yang membaca data uji, dan hanya sekali.

    Args:
        config: Konfigurasi project.
        logger: Logger yang dipakai.
        schemes: Skema yang dijalankan.
        horizon: Panjang horizon H.

    Returns:
        Keluaran :func:`src.forecasting.run_all_schemes`.
    """
    logger.warning(
        "Tahap ini membaca DATA UJI. Untuk mencegah kebocoran data, hasilnya hanya boleh "
        "dilaporkan apa adanya — jangan mengulang tahap ini demi angka yang lebih baik."
    )

    return run_all_schemes(
        config=config,
        schemes=schemes,
        split="test",
        horizon=horizon,
        logger=logger,
    )


# =============================================================================
# Tahap 5 — evaluasi dan tabel BAB IV
# =============================================================================


def collect_forecasts(
    config: dict[str, Any], horizon: int, logger: logging.Logger
) -> dict[str, dict[str, Any]]:
    """Memuat seluruh ramalan uji yang tersedia dari berkas parquet.

    Membaca dari disk, bukan dari objek hasil tahap 4, agar tahap evaluasi dapat
    dijalankan ulang tanpa menjalankan model lagi — dan agar yang dievaluasi
    benar-benar angka yang tersimpan pada laporan.

    Args:
        config: Konfigurasi project.
        horizon: Panjang horizon H.
        logger: Logger yang dipakai.

    Returns:
        Dictionary ``{skema: array ramalan}``.

    Raises:
        PipelineError: Bila tidak ada satu pun berkas ramalan.
    """
    forecasts: dict[str, dict[str, Any]] = {}

    for scheme in ALL_SCHEMES:
        path = forecast_file_path(scheme, horizon, "test", config)
        if path.is_file():
            forecasts[scheme] = load_forecast_arrays(scheme, horizon, "test", config)
            logger.info("Memuat ramalan '%s' dari %s", scheme, path.name)
        else:
            logger.warning("Ramalan '%s' tidak ditemukan, skema dilewati.", scheme)

    if not forecasts:
        raise PipelineError(
            f"Tidak ada berkas ramalan untuk H={horizon} pada periode uji. "
            f"Jalankan tahap 4 terlebih dahulu."
        )

    # Seluruh skema WAJIB berpijak pada jendela yang identik
    origin_sets = {scheme: tuple(f["origins"]) for scheme, f in forecasts.items()}
    unique_origins = set(origin_sets.values())
    if len(unique_origins) > 1:
        raise PipelineError(
            f"Skema tidak memakai daftar origin yang sama — protokol rolling origin dilanggar. "
            f"Jumlah origin per skema: "
            f"{ {scheme: len(origins) for scheme, origins in origin_sets.items()} }"
        )

    return forecasts


def compute_all_metrics(
    forecasts: dict[str, dict[str, Any]], config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Menghitung metrik lengkap untuk tiap skema.

    Penyebut MASE selalu diambil dari baseline naive persistence pada jendela
    yang sama, sehingga angka MASE seluruh skema dapat langsung diperbandingkan.

    Args:
        forecasts: Dictionary ``{skema: array ramalan}``.
        config: Konfigurasi project.

    Returns:
        Dictionary ``{skema: keluaran aggregate_metrics}``.
    """
    quantile_levels = config["model"]["quantile_levels"]
    naive_point = forecasts.get(SCHEME_NAIVE, {}).get("y_pred")

    return {
        scheme: aggregate_metrics(
            forecast["y_true"],
            forecast["y_pred_quantiles"],
            quantile_levels,
            y_naive=naive_point,
            config=config,
        )
        for scheme, forecast in forecasts.items()
    }


def compute_dm_tests(
    forecasts: dict[str, dict[str, Any]], horizon: int, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Menjalankan uji Diebold-Mariano untuk setiap pasangan skema.

    Rugi yang diuji adalah galat absolut, konsisten dengan MAE dan MASE yang
    dilaporkan. Koreksi HAC Newey-West dengan lag ``H - 1`` wajib dipakai karena
    jendela rolling origin saling tumpang tindih.

    Args:
        forecasts: Dictionary ``{skema: array ramalan}``.
        horizon: Panjang horizon H.
        config: Konfigurasi project.

    Returns:
        Daftar hasil uji, satu entri per pasangan yang tersedia.
    """
    alpha = config["evaluation"]["diebold_mariano"]["alpha"]
    tests: list[dict[str, Any]] = []

    for scheme_a, scheme_b, is_primary in DM_PAIRS:
        if scheme_a not in forecasts or scheme_b not in forecasts:
            continue

        loss_a = np.abs(forecasts[scheme_a]["y_true"] - forecasts[scheme_a]["y_pred"])
        loss_b = np.abs(forecasts[scheme_b]["y_true"] - forecasts[scheme_b]["y_pred"])
        result = diebold_mariano(loss_a, loss_b, h=horizon, alpha=alpha)

        winner = {"a": scheme_a, "b": scheme_b, "seri": None}[result["better"]]
        result.update(
            {
                "scheme_a": scheme_a,
                "scheme_b": scheme_b,
                "primary": is_primary,
                "loss": "galat absolut",
                "better_scheme": winner,
                "label": (
                    f"{SCHEME_DISPLAY.get(scheme_a, scheme_a)} vs "
                    f"{SCHEME_DISPLAY.get(scheme_b, scheme_b)}"
                ),
            }
        )
        tests.append(result)

    return tests


def _format_number(value: Any, digits: int = 4) -> str:
    """Memformat sebuah angka untuk tabel markdown.

    Args:
        value: Nilai yang diformat; ``None`` menjadi ``"—"``.
        digits: Banyaknya angka di belakang koma.

    Returns:
        String siap tempel ke tabel.
    """
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def build_comparison_table(
    metrics: dict[str, dict[str, Any]],
    dm_tests: list[dict[str, Any]],
    protocol: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """Menyusun tabel perbandingan markdown yang siap disalin ke BAB IV.

    Baris = skema, kolom = seluruh metrik yang disyaratkan penelitian, disusul
    tabel kedua berisi hasil uji Diebold-Mariano. Angka sengaja tidak diberi
    penanda "terbaik" secara otomatis: perbedaan yang tidak signifikan secara
    statistik tidak layak ditampilkan seolah-olah menang.

    Args:
        metrics: Dictionary ``{skema: keluaran aggregate_metrics}``.
        dm_tests: Keluaran :func:`compute_dm_tests`.
        protocol: Ringkasan protokol evaluasi (H, stride, jumlah jendela, dst).
        config: Konfigurasi project.

    Returns:
        Isi berkas ``comparison_table.md`` sebagai string.
    """
    coverage_keys = [f"{level:g}" for level in config["evaluation"]["coverage_levels"]]
    low_key = coverage_keys[0] if coverage_keys else "0.8"
    high_key = coverage_keys[1] if len(coverage_keys) > 1 else "0.95"

    lines: list[str] = [
        "# Tabel Perbandingan Skema — BAB IV",
        "",
        f"Protokol: rolling origin (expanding window), H = {protocol['horizon']} hari "
        f"perdagangan, stride = {protocol['stride']}, "
        f"{protocol['n_windows']} jendela pada periode uji "
        f"({protocol['test_start']} s.d. {protocol['test_end']}).",
        "",
        "Seluruh skema dievaluasi pada **himpunan jendela rolling origin yang persis "
        "sama**. Kovariat Emas dan Dollar Index diperlakukan sebagai *past-only*: "
        "nilai masa depannya tidak pernah diberikan ke model.",
        "",
        "## Tabel 4.x — Perbandingan akurasi dan kalibrasi",
        "",
        "| Skema | MAE | RMSE | MAPE (%) | MASE | CRPS | nCRPS | Cov80 | Cov95 | Width80 | CrossRate |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for scheme in TABLE_ORDER:
        if scheme not in metrics:
            # Skema yang belum dijalankan tetap ditampilkan sebagai baris kosong.
            # Menghilangkannya diam-diam akan membuat tabel tampak lengkap
            # padahal ada skema yang belum diuji.
            lines.append(
                f"| {SCHEME_DISPLAY.get(scheme, scheme)} | "
                + " | ".join(["_belum dijalankan_"] + ["—"] * 9)
                + " |"
            )
            continue
        overall = metrics[scheme]["overall"]
        lines.append(
            "| {name} | {mae} | {rmse} | {mape} | {mase} | {crps} | {ncrps} | "
            "{cov80} | {cov95} | {width80} | {cross} |".format(
                name=SCHEME_DISPLAY.get(scheme, scheme),
                mae=_format_number(overall["mae"]),
                rmse=_format_number(overall["rmse"]),
                mape=_format_number(overall["mape"], 3),
                mase=_format_number(overall.get("mase")),
                crps=_format_number(overall["crps"]),
                ncrps=_format_number(overall["normalized_crps"], 6),
                cov80=_format_number(overall["coverage"].get(low_key), 4),
                cov95=_format_number(overall["coverage"].get(high_key), 4),
                width80=_format_number(overall["interval_width"].get(low_key)),
                cross=_format_number(overall["quantile_crossing_rate"], 6),
            )
        )

    lines += [
        "",
        "Keterangan kolom: MASE = MAE skema / MAE baseline pada jendela yang sama "
        "(< 1 berarti mengungguli baseline). nCRPS = CRPS / rata-rata |y|. "
        "Cov80 dan Cov95 = cakupan empiris interval 80% dan 95% (target 0,80 dan 0,95). "
        "Width80 = lebar rata-rata interval 80% dalam USD — wajib dibaca bersama Cov80, "
        "karena cakupan tinggi yang dicapai lewat interval kelewat lebar bukan kalibrasi "
        "yang baik. CrossRate = proporsi pelanggaran urutan kuantil.",
        "",
        "## Tabel 4.y — Uji Diebold-Mariano (rugi = galat absolut, HAC Newey-West)",
        "",
        "| Perbandingan | Selisih rugi rata-rata | Statistik DM | p-value | Lag HAC | Signifikan (α = {alpha:g}) | Lebih baik |".format(
            alpha=config["evaluation"]["diebold_mariano"]["alpha"]
        ),
        "|---|---|---|---|---|---|---|",
    ]

    # Uji utama yang belum dapat dijalankan tetap dicatat sebagai baris tertunda,
    # supaya pembaca tabel tahu ada uji yang masih hilang.
    executed_pairs = {(test["scheme_a"], test["scheme_b"]) for test in dm_tests}
    for scheme_a, scheme_b, is_primary in DM_PAIRS:
        if (scheme_a, scheme_b) in executed_pairs:
            continue
        label = (
            f"{SCHEME_DISPLAY.get(scheme_a, scheme_a)} vs "
            f"{SCHEME_DISPLAY.get(scheme_b, scheme_b)}"
        )
        marker = " **(uji utama)**" if is_primary else ""
        lines.append(
            f"| {label}{marker} | "
            + " | ".join(["_belum dijalankan_"] + ["—"] * 5)
            + " |"
        )

    for test in dm_tests:
        better = (
            "seri (tidak berbeda)"
            if test["better_scheme"] is None
            else SCHEME_DISPLAY.get(test["better_scheme"], test["better_scheme"])
        )
        marker = " **(uji utama)**" if test["primary"] else ""
        lines.append(
            "| {label}{marker} | {diff} | {stat} | {p} | {lag} | {sig} | {better} |".format(
                label=test["label"],
                marker=marker,
                diff=_format_number(test["mean_loss_differential"], 6),
                stat=_format_number(test["statistic"], 4),
                p=_format_number(test["p_value"], 4),
                lag=test["hac_lag"],
                sig="ya" if test["significant"] else "tidak",
                better=better if test["significant"] else "tidak terbukti berbeda",
            )
        )

    lines += [
        "",
        "H0 uji DM: kedua ramalan sama akuratnya. Statistik negatif berarti skema "
        "pertama memiliki rugi lebih kecil. Karena jendela rolling origin saling "
        f"tumpang tindih, ragam diestimasi dengan HAC Newey-West berlag H − 1 = "
        f"{protocol['horizon'] - 1}. **p-value di atas α berarti perbedaan angka pada "
        "tabel 4.x tidak terbukti secara statistik dan tidak boleh dinarasikan sebagai "
        "keunggulan.**",
        "",
    ]

    return "\n".join(lines)


def stage_evaluation(
    config: dict[str, Any], logger: logging.Logger, horizon: int
) -> dict[str, Any]:
    """Tahap 5 — metrik lengkap, uji DM, ``final_results.json``, tabel BAB IV.

    Args:
        config: Konfigurasi project.
        logger: Logger yang dipakai.
        horizon: Panjang horizon H.

    Returns:
        Dictionary berisi ``forecasts``, ``metrics``, ``dm_tests``, dan
        ``final_results``.
    """
    forecasts = collect_forecasts(config, horizon, logger)
    metrics = compute_all_metrics(forecasts, config)
    dm_tests = compute_dm_tests(forecasts, horizon, config)

    reference = next(iter(forecasts.values()))
    origin_dates = reference["origin_dates"]
    protocol = {
        "horizon": int(horizon),
        "stride": int(config["evaluation"]["stride"]),
        "max_context": int(config["model"]["max_context"]),
        "n_windows": int(reference["y_true"].shape[0]),
        "test_start": str(origin_dates[0].date()),
        "test_end": str(origin_dates[-1].date()),
        "quantile_levels": list(config["model"]["quantile_levels"]),
        "known_covariates_names": list(config["model"]["known_covariates_names"]),
        "context_type": "expanding (seluruh riwayat sampai origin, dibatasi max_context)",
        "schemes_evaluated": list(forecasts),
    }

    # --- Ringkasan tuning fine-tuning, bila tahap itu memang dijalankan ---
    tuning_path = resolve_path(config["paths"]["metrics_dir"]) / "finetune_tuning.json"
    finetune_summary: dict[str, Any] | None = None
    if tuning_path.is_file():
        import json

        with tuning_path.open("r", encoding="utf-8") as file:
            tuning = json.load(file)
        finetune_summary = {
            "best_params": tuning.get("best_params"),
            "best_val_score": tuning.get("best_score"),
            "selection_metric": tuning.get("selection_metric"),
            "selection_split": tuning.get("selection_split"),
            "n_candidates": tuning.get("n_candidates"),
            "n_failed": tuning.get("n_failed"),
        }

    final_results: dict[str, Any] = {
        "protocol": protocol,
        "finetune_selection": finetune_summary,
        "metrics": metrics,
        "diebold_mariano": dm_tests,
    }

    metrics_dir = resolve_path(config["paths"]["metrics_dir"])
    results_path = save_json(final_results, metrics_dir / "final_results.json")
    logger.info("Hasil akhir tersimpan di %s", results_path)

    table = build_comparison_table(metrics, dm_tests, protocol, config)
    table_path = metrics_dir / "comparison_table.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(table, encoding="utf-8")
    logger.info("Tabel perbandingan tersimpan di %s", table_path)

    _log_metrics_summary(metrics, dm_tests, logger)

    return {
        "forecasts": forecasts,
        "metrics": metrics,
        "dm_tests": dm_tests,
        "final_results": final_results,
        "table_path": table_path,
    }


def _log_metrics_summary(
    metrics: dict[str, dict[str, Any]],
    dm_tests: list[dict[str, Any]],
    logger: logging.Logger,
) -> None:
    """Mencetak ringkasan metrik dan uji DM ke log.

    Args:
        metrics: Dictionary ``{skema: keluaran aggregate_metrics}``.
        dm_tests: Keluaran :func:`compute_dm_tests`.
        logger: Logger tujuan.
    """
    logger.info("-" * 78)
    logger.info(
        "%-22s %8s %8s %8s %8s %8s %8s",
        "SKEMA", "MAE", "RMSE", "MASE", "CRPS", "Cov80", "Cov95",
    )
    for scheme in TABLE_ORDER:
        if scheme not in metrics:
            continue
        overall = metrics[scheme]["overall"]
        logger.info(
            "%-22s %8.4f %8.4f %8s %8.4f %8.4f %8.4f",
            SCHEME_DISPLAY.get(scheme, scheme),
            overall["mae"],
            overall["rmse"],
            _format_number(overall.get("mase")),
            overall["crps"],
            overall["coverage"].get("0.8", float("nan")),
            overall["coverage"].get("0.95", float("nan")),
        )

    logger.info("-" * 78)
    for test in dm_tests:
        logger.info(
            "DM %-46s stat=%7.4f  p=%.4f  %s",
            test["label"] + (" (utama)" if test["primary"] else ""),
            test["statistic"],
            test["p_value"],
            "SIGNIFIKAN" if test["significant"] else "tidak signifikan",
        )
    logger.info("-" * 78)


# =============================================================================
# Tahap 6 — visualisasi
# =============================================================================


def stage_visualization(
    config: dict[str, Any],
    logger: logging.Logger,
    forecasts: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    """Tahap 6 — menghasilkan seluruh gambar laporan.

    Args:
        config: Konfigurasi project.
        logger: Logger yang dipakai.
        forecasts: Dictionary ``{skema: array ramalan}``.
        metrics: Dictionary ``{skema: keluaran aggregate_metrics}``.

    Returns:
        Dictionary ``{nama_gambar: path}``.
    """
    frames = load_processed_frames(
        ("train", "val", "test"), config=config, logger=logger
    )

    return generate_all_figures(
        forecasts, metrics=metrics, frames=frames, config=config, logger=logger
    )


# =============================================================================
# Titik masuk
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Membaca argumen baris perintah pipeline.

    Args:
        argv: Daftar argumen. Bila ``None``, diambil dari ``sys.argv``.

    Returns:
        Namespace berisi ``skip_download``, ``horizon``, dan ``scheme``.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline peramalan probabilistik harga Perak dengan Chronos-2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Contoh:\n"
            "  python run_pipeline.py\n"
            "  python run_pipeline.py --skip-download\n"
            "  python run_pipeline.py --horizon 5 --scheme zeroshot naive\n"
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Lewati tahap 1; pakai berkas mentah di data/raw/ apa adanya.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Panjang horizon H. Bawaan: model.prediction_length pada config.",
    )
    parser.add_argument(
        "--scheme",
        nargs="+",
        choices=[*ALL_SCHEMES, "all"],
        default=["all"],
        help="Skema yang dijalankan pada tahap 4 (bawaan: all).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Lewati tahap 1-4; susun ulang tabel dan gambar dari berkas ramalan "
            "yang sudah ada di results/forecasts/. Dipakai setelah menyalin "
            "hasil fine-tuning dari mesin lain (mis. Colab GPU)."
        ),
    )
    return parser.parse_args(argv)


def _run_report_only(
    config: dict[str, Any],
    logger: logging.Logger,
    horizon: int,
    report: dict[str, Any],
    durations: dict[str, float],
    pipeline_started_at: float,
) -> dict[str, Any]:
    """Menjalankan hanya tahap 5 dan 6 dari berkas ramalan yang sudah ada.

    Dipakai pada alur kerja dua mesin: fine-tuning dijalankan di GPU (mis.
    Colab), berkas ``results/forecasts/finetuned_H{H}.parquet`` disalin ke mesin
    ini, lalu tabel dan gambar disusun ulang tanpa menjalankan model apa pun.
    Karena tidak ada model yang dijalankan, data uji tidak diramalkan ulang.

    Args:
        config: Konfigurasi project.
        logger: Logger yang dipakai.
        horizon: Panjang horizon H.
        report: Dictionary laporan yang sedang disusun.
        durations: Dictionary durasi tiap tahap.
        pipeline_started_at: Waktu mulai pipeline.

    Returns:
        Dictionary laporan pipeline.
    """
    started = _log_stage(logger, 5, "Evaluasi & tabel BAB IV")
    evaluation_result = stage_evaluation(config, logger, horizon)
    report["stage_5_evaluation"] = {
        "final_results": str(
            resolve_path(config["paths"]["metrics_dir"]) / "final_results.json"
        ),
        "comparison_table": str(evaluation_result["table_path"]),
    }
    durations["stage_5_evaluation"] = _log_stage_done(
        logger, 5, "Evaluasi & tabel BAB IV", started
    )

    started = _log_stage(logger, 6, "Visualisasi")
    figures = stage_visualization(
        config, logger, evaluation_result["forecasts"], evaluation_result["metrics"]
    )
    report["stage_6_visualization"] = {name: str(path) for name, path in figures.items()}
    durations["stage_6_visualization"] = _log_stage_done(logger, 6, "Visualisasi", started)

    report["durations_seconds"] = durations
    report["total_seconds"] = time.perf_counter() - pipeline_started_at
    report["mode"] = "report-only"
    report["evaluation"] = evaluation_result

    logger.info("")
    logger.info("#" * 78)
    logger.info(
        "LAPORAN DISUSUN ULANG dalam %.1f detik. Tabel BAB IV: %s",
        report["total_seconds"],
        evaluation_result["table_path"],
    )
    logger.info("#" * 78)

    return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Menjalankan seluruh pipeline dari tahap 1 sampai 6.

    Args:
        argv: Daftar argumen baris perintah.

    Returns:
        Dictionary berisi status tiap tahap, durasinya, dan hasil akhir.
    """
    args = parse_args(argv)

    config = load_config()
    logger = setup_logger(
        "pipeline", log_dir=config.get("paths", {}).get("logs_dir", "logs")
    )

    horizon = args.horizon or config["model"]["prediction_length"]
    schemes = (
        ALL_SCHEMES if "all" in args.scheme else tuple(dict.fromkeys(args.scheme))
    )

    logger.info("#" * 78)
    logger.info("PIPELINE PERAMALAN HARGA PERAK DENGAN CHRONOS-2")
    logger.info("H = %d | skema = %s | skip-download = %s", horizon, list(schemes), args.skip_download)
    logger.info("#" * 78)

    log_environment(config=config, logger=logger)

    durations: dict[str, float] = {}
    report: dict[str, Any] = {"horizon": horizon, "schemes": list(schemes)}
    pipeline_started_at = time.perf_counter()

    if args.report_only:
        logger.warning(
            "--report-only: tahap 1-4 dilewati. Tabel dan gambar disusun ulang "
            "dari berkas ramalan yang sudah ada."
        )
        return _run_report_only(
            config, logger, horizon, report, durations, pipeline_started_at
        )

    started = _log_stage(logger, 1, "Pengumpulan data")
    report["stage_1_data_collection"] = stage_data_collection(
        config, logger, args.skip_download
    )
    durations["stage_1_data_collection"] = _log_stage_done(
        logger, 1, "Pengumpulan data", started
    )

    started = _log_stage(logger, 2, "Preprocessing & pembagian data")
    report["stage_2_preprocessing"] = stage_preprocessing(config, logger)
    durations["stage_2_preprocessing"] = _log_stage_done(
        logger, 2, "Preprocessing & pembagian data", started
    )

    started = _log_stage(logger, 3, "Mutual information")
    report["stage_3_mutual_information"] = stage_mutual_information(config, logger)
    durations["stage_3_mutual_information"] = _log_stage_done(
        logger, 3, "Mutual information", started
    )

    started = _log_stage(logger, 4, "Peramalan Chronos-2")
    forecasting_result = stage_forecasting(config, logger, schemes, horizon)
    report["stage_4_forecasting"] = forecasting_result["summary"]
    durations["stage_4_forecasting"] = _log_stage_done(
        logger, 4, "Peramalan Chronos-2", started
    )

    started = _log_stage(logger, 5, "Evaluasi & tabel BAB IV")
    evaluation_result = stage_evaluation(config, logger, horizon)
    report["stage_5_evaluation"] = {
        "final_results": str(
            resolve_path(config["paths"]["metrics_dir"]) / "final_results.json"
        ),
        "comparison_table": str(evaluation_result["table_path"]),
    }
    durations["stage_5_evaluation"] = _log_stage_done(
        logger, 5, "Evaluasi & tabel BAB IV", started
    )

    started = _log_stage(logger, 6, "Visualisasi")
    figures = stage_visualization(
        config,
        logger,
        evaluation_result["forecasts"],
        evaluation_result["metrics"],
    )
    report["stage_6_visualization"] = {name: str(path) for name, path in figures.items()}
    durations["stage_6_visualization"] = _log_stage_done(logger, 6, "Visualisasi", started)

    total_seconds = time.perf_counter() - pipeline_started_at
    report["durations_seconds"] = durations
    report["total_seconds"] = total_seconds

    save_json(
        report,
        resolve_path(config["paths"]["metrics_dir"]) / f"pipeline_report_H{horizon}.json",
    )

    logger.info("")
    logger.info("#" * 78)
    logger.info("PIPELINE SELESAI dalam %.1f detik (%.1f menit)", total_seconds, total_seconds / 60)
    for stage_name, seconds in durations.items():
        logger.info("  %-28s %8.1f detik", stage_name, seconds)
    logger.info("Tabel BAB IV: %s", evaluation_result["table_path"])
    logger.info("#" * 78)

    report["evaluation"] = evaluation_result

    return report


if __name__ == "__main__":
    main()
