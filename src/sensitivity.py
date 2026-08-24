"""Analisis sensitivitas: horizon, ablasi kovariat, dan skenario oracle/ex-post.

Modul ini dijalankan **setelah**
hasil utama (H=10, `results/metrics/final_results.json`) stabil. Seluruh keluaran
disimpan ke berkas terpisah (`results/metrics/sensitivity_*.json`,
`results/forecasts/{scheme}_H{h}.parquet` dengan h != 10 atau nama skema berawalan
``ablation_``/``oracle_expost``) sehingga tidak pernah tercampur dengan hasil utama.

Tiga analisis:

1. :func:`run_horizon_sensitivity` -- mengulang protokol rolling origin pada
   H = 5 dan H = 20 (nilai bawaan dari ``evaluation.horizons_sensitivity``),
   membandingkan naive/zero-shot/fine-tuned seperti hasil utama.
2. :func:`run_covariate_ablation` -- membandingkan tiga kondisi kovariat input
   (hanya Perak; Perak+Emas; Perak+Emas+DXY) pada skema **terbaik** hasil utama,
   untuk menguji secara empiris apakah kovariat benar-benar membantu, disambungkan
   dengan temuan Mutual Information (`results/metrics/mutual_information.json`).
3. :func:`run_oracle_ablation` -- **opsional**, kovariat diperlakukan sebagai
   known-future dengan nilai AKTUALnya (bukan skenario realistis, karena nilai
   masa depan kovariat tidak diketahui). Hasilnya HANYA batas atas teoretis dan diberi label
   **"EX-POST / TIDAK REALISTIS"** di setiap output.

Tuning grid fine-tuning berbiaya besar dan berlipat per horizon: satu grid penuh
berisi 18 kandidat (terukur ~21 menit di GPU lokal dan ~27 menit di Colab GPU untuk
H=10; berjam-jam di CPU), sehingga men-tuning ulang tiap horizon sensitivitas
berarti mengulang seluruh pencarian itu sebanyak jumlah horizon. Karena itu
:func:`run_horizon_sensitivity` menyediakan opsi ``reuse_finetune_params`` untuk
memakai ulang hyperparameter terbaik hasil tuning H=10 tanpa tuning ulang per
horizon -- penyimpangan dari protokol pemilihan hyperparameter pada horizon utama
yang WAJIB dicatat sebagai catatan metodologis bila dipakai (lihat argumen fungsi).

Cara menjalankan mandiri:
    python -m src.sensitivity                                  # ketiga analisis
    python -m src.sensitivity --skip-oracle
    python -m src.sensitivity --skip-covariate --skip-oracle
    python -m src.sensitivity --horizons 5 20 --reuse-finetune-params
    python -m src.sensitivity --skip-finetune-horizon           # horizon: naive+zero-shot saja
    python -m src.sensitivity --summary-only                    # hanya susun ulang md+figure
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_collection import DATE_INDEX_NAME  # noqa: E402
from src.evaluation import aggregate_metrics, diebold_mariano  # noqa: E402
from src.forecasting import (  # noqa: E402
    SCHEME_FINETUNED,
    SCHEME_NAIVE,
    SCHEME_ZEROSHOT,
    build_finetuned_predictor,
    build_oracle_predictor,
    build_origins,
    build_zeroshot_predictor,
    load_forecast_arrays,
    load_processed_frames,
    naive_rolling_forecast,
    rolling_forecast,
    save_forecast,
    to_model_tsdf,
    tune_finetune_hyperparams,
)
from src.preprocessing import get_covariate_columns
from src.utils import load_config, resolve_path, save_json, setup_logger

# Label dan warna tampil -- identik dengan src/visualization.py agar seluruh
# gambar laporan konsisten satu sama lain.
SCHEME_LABELS: dict[str, str] = {
    SCHEME_NAIVE: "Naive Persistence",
    SCHEME_ZEROSHOT: "Chronos-2 Zero-Shot",
    SCHEME_FINETUNED: "Chronos-2 Fine-Tuned",
}
SCHEME_COLORS: dict[str, str] = {
    SCHEME_NAIVE: "#898781",
    SCHEME_ZEROSHOT: "#2a78d6",
    SCHEME_FINETUNED: "#eb6834",
}
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRIDLINE = "#e1e0d9"
_BASELINE = "#c3c2b7"
_ORACLE_COLOR = "#b23a48"  # merah peringatan -- khusus skenario ex-post

# Label kondisi ablasi kovariat, urut dari sedikit ke banyak kovariat
CONDITION_LABELS: dict[str, str] = {
    "silver_only": "Hanya Perak",
    "silver_gold": "Perak + Emas",
    "silver_gold_dxy": "Perak + Emas + Dollar Index",
}

ORACLE_LABEL = (
    "EX-POST / TIDAK REALISTIS -- batas atas teoretis. Kovariat Emas/DXY diberi "
    "nilai AKTUALnya pada horizon ramalan (known-future), sesuatu yang TIDAK "
    "tersedia saat peramalan sungguhan dilakukan. "
    "JANGAN dinarasikan sebagai keunggulan model yang dapat dipakai."
)


class SensitivityError(RuntimeError):
    """Kegagalan pada analisis sensitivitas."""


# =============================================================================
# Utilitas bersama
# =============================================================================


def _format_number(value: Any, digits: int = 4) -> str:
    """Memformat angka untuk tabel markdown; ``None`` menjadi ``"—"``."""
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Memuat berkas JSON bila ada, atau ``None`` bila belum pernah dijalankan."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _determine_best_scheme(config: dict[str, Any], logger: logging.Logger) -> str:
    """Menentukan skema Chronos-2 terbaik dari hasil utama (H=10).

    "Terbaik" didefinisikan sebagai *normalized CRPS* (nCRPS) terkecil --
    metrik yang sama dengan ``finetune.selection_metric`` ("WQL", yang secara
    eksak sama dengan nCRPS/2, lihat :func:`src.forecasting.weighted_quantile_loss`)
    -- dibandingkan antara zero-shot dan fine-tuned. Ablasi kovariat dan skenario
    oracle dijalankan pada skema ini, bukan pada keduanya, agar biayanya tetap
    terkendali.

    Args:
        config: Konfigurasi project.
        logger: Logger yang dipakai.

    Returns:
        Nama skema terbaik (``"zeroshot"`` atau ``"finetuned"``). Bawaan
        ``"zeroshot"`` bila ``final_results.json`` belum ada.
    """
    final_results_path = resolve_path(config["paths"]["metrics_dir"]) / "final_results.json"
    if not final_results_path.is_file():
        logger.warning(
            "final_results.json tidak ditemukan; memakai '%s' sebagai skema terbaik bawaan.",
            SCHEME_ZEROSHOT,
        )
        return SCHEME_ZEROSHOT

    with final_results_path.open("r", encoding="utf-8") as file:
        final_results = json.load(file)

    candidates = [
        scheme
        for scheme in (SCHEME_ZEROSHOT, SCHEME_FINETUNED)
        if scheme in final_results.get("metrics", {})
    ]
    if not candidates:
        return SCHEME_ZEROSHOT

    scores = {
        scheme: final_results["metrics"][scheme]["overall"]["normalized_crps"]
        for scheme in candidates
    }
    best = min(scores, key=scores.get)
    logger.info(
        "Skema terbaik ditentukan dari final_results.json (nCRPS terkecil): %s -> '%s'",
        {scheme: round(score, 6) for scheme, score in scores.items()},
        best,
    )
    return best


def _load_mutual_information_crossref(
    config: dict[str, Any], logger: logging.Logger
) -> dict[str, Any] | None:
    """Memuat ringkasan Mutual Information untuk disambungkan ke ablasi kovariat.

    Args:
        config: Konfigurasi project.
        logger: Logger yang dipakai.

    Returns:
        Dictionary ``{"source": ..., "variants": {mode: {kovariat: mi_mean}}}``,
        atau ``None`` bila berkasnya belum ada.
    """
    mi_path = resolve_path(config["paths"]["metrics_dir"]) / "mutual_information.json"
    if not mi_path.is_file():
        logger.warning("mutual_information.json tidak ditemukan, crossref MI dilewati.")
        return None

    with mi_path.open("r", encoding="utf-8") as file:
        mi = json.load(file)

    variants: dict[str, dict[str, float]] = {}
    for mode, entry in mi.get("results", {}).items():
        variants[mode] = {
            covariate: float(pair["mi_mean"]) for covariate, pair in entry.get("pairs", {}).items()
        }

    return {"source": "results/metrics/mutual_information.json", "variants": variants}


def _run_dm_pairs(
    forecasts: dict[str, dict[str, Any]],
    pairs: list[tuple[str, str, bool]],
    horizon: int,
    alpha: float,
    labels: dict[str, str],
    key_a: str = "scheme_a",
    key_b: str = "scheme_b",
    better_key: str = "better_scheme",
) -> list[dict[str, Any]]:
    """Menjalankan uji Diebold-Mariano untuk sederet pasangan (rugi = galat absolut).

    Args:
        forecasts: Dictionary ``{nama: keluaran rolling_forecast}``.
        pairs: Daftar ``(nama_a, nama_b, is_primary)``.
        horizon: Panjang horizon H (menentukan lag HAC = H - 1).
        alpha: Taraf nyata.
        labels: Label tampil tiap nama.
        key_a/key_b/better_key: Nama field pada hasil, agar dapat dipakai baik
            untuk pasangan skema maupun pasangan kondisi ablasi.

    Returns:
        Daftar hasil uji, satu entri per pasangan yang tersedia pada ``forecasts``.
    """
    results: list[dict[str, Any]] = []
    for name_a, name_b, is_primary in pairs:
        if name_a not in forecasts or name_b not in forecasts:
            continue
        loss_a = np.abs(forecasts[name_a]["y_true"] - forecasts[name_a]["y_pred"])
        loss_b = np.abs(forecasts[name_b]["y_true"] - forecasts[name_b]["y_pred"])
        result = diebold_mariano(loss_a, loss_b, h=horizon, alpha=alpha)
        winner = {"a": name_a, "b": name_b, "seri": None}[result["better"]]
        result.update(
            {
                key_a: name_a,
                key_b: name_b,
                "primary": is_primary,
                "loss": "galat absolut",
                better_key: winner,
                "label": f"{labels.get(name_a, name_a)} vs {labels.get(name_b, name_b)}",
            }
        )
        results.append(result)
    return results


# =============================================================================
# 1. Sensitivitas horizon
# =============================================================================


def run_horizon_sensitivity(
    horizons: list[int] | None = None,
    include_finetune: bool = True,
    reuse_finetune_params: bool = False,
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Mengulang protokol rolling origin pada horizon selain H utama.

    Setiap horizon punya daftar origin sendiri (jumlah jendela berbeda dari
    H=10), sehingga tidak bisa dibandingkan jendela-demi-jendela lintas horizon
    -- yang dibandingkan adalah metrik teragregasi. Naive dan zero-shot selalu
    dijalankan (murah, tanpa pelatihan). Fine-tuned bersifat opsional lewat
    ``include_finetune`` karena tuning grid penuh harus diulang untuk tiap
    horizon.

    Args:
        horizons: Daftar horizon yang diuji. Bila ``None``, diambil dari
            ``evaluation.horizons_sensitivity`` config (``[5, 20]``).
        include_finetune: Bila ``False``, skema fine-tuned dilewati sepenuhnya
            pada horizon-horizon ini (naive dan zero-shot tetap dijalankan).
        reuse_finetune_params: Bila ``True`` DAN ``include_finetune=True``,
            fine-tuning pada tiap horizon memakai ULANG hyperparameter terbaik
            hasil tuning H utama (``results/metrics/finetune_tuning.json``)
            tanpa tuning grid ulang -- penyimpangan eksplisit dari protokol
            pemilihan hyperparameter
            demi kelayakan komputasi, dicatat pada ``finetune_selection.retuned``.
            Bila ``False``, tuning grid penuh dijalankan ulang per horizon
            (mahal, lihat docstring modul).
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dictionary ``{"protocol": {...}, "results": {str(H): {...}}}``, juga
        disimpan ke ``results/metrics/sensitivity_horizon.json``. Horizon utama
        (H=10) disertakan sebagai baris pembanding, dimuat apa adanya dari
        ``final_results.json`` -- TIDAK dihitung ulang.

    Raises:
        SensitivityError: Bila ``reuse_finetune_params=True`` tetapi
            ``finetune_tuning.json`` belum ada.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("sensitivity", log_dir=config.get("paths", {}).get("logs_dir", "logs"))
    if horizons is None:
        horizons = list(config["evaluation"]["horizons_sensitivity"])

    reference_horizon = config["model"]["prediction_length"]
    stride = config["evaluation"]["stride"]
    max_context = config["model"]["max_context"]
    quantile_levels = config["model"]["quantile_levels"]
    alpha = config["evaluation"]["diebold_mariano"]["alpha"]

    logger.info("=" * 78)
    logger.info(
        "SENSITIVITAS HORIZON -- H=%s (H utama=%d, disimpan terpisah dari hasil utama)",
        horizons,
        reference_horizon,
    )
    logger.info("=" * 78)

    frames = load_processed_frames(("train", "val", "test"), config=config, logger=logger)
    full_df = pd.concat([frames["train"], frames["val"], frames["test"]])
    full_df.index.name = DATE_INDEX_NAME
    eval_start_idx = len(frames["train"]) + len(frames["val"])

    reused_best_params: dict[str, Any] | None = None
    if include_finetune and reuse_finetune_params:
        tuning_path = resolve_path(config["paths"]["metrics_dir"]) / "finetune_tuning.json"
        if not tuning_path.is_file():
            raise SensitivityError(
                f"reuse_finetune_params=True tetapi '{tuning_path}' tidak ditemukan. "
                "Jalankan tahap 4 pipeline utama (H utama) terlebih dahulu."
            )
        with tuning_path.open("r", encoding="utf-8") as file:
            reused_best_params = json.load(file)["best_params"]
        logger.warning(
            "Fine-tuning pada H=%s memakai ULANG hyperparameter terbaik hasil tuning "
            "H=%d (%s) TANPA tuning grid ulang per horizon -- penyimpangan eksplisit "
            "dari protokol pemilihan hyperparameter demi kelayakan komputasi, karena "
            "tuning ulang berarti mengulang seluruh pencarian 18 kandidat untuk tiap "
            "horizon (lihat docstring modul).",
            horizons,
            reference_horizon,
            reused_best_params,
        )

    results_per_horizon: dict[str, Any] = {}

    for horizon in horizons:
        logger.info("-" * 78)
        logger.info("HORIZON SENSITIVITAS H=%d", horizon)
        logger.info("-" * 78)

        origins = build_origins(len(full_df), eval_start_idx, horizon, stride)
        train_tsdf = to_model_tsdf(frames["train"], config)

        forecasts: dict[str, dict[str, Any]] = {}

        naive_forecast = naive_rolling_forecast(
            full_df, origins, horizon, max_context, config=config, logger=logger
        )
        assert naive_forecast["origins"] == list(origins)
        save_forecast(naive_forecast, SCHEME_NAIVE, horizon, "test", config, logger)
        forecasts[SCHEME_NAIVE] = naive_forecast

        zeroshot_predictor = build_zeroshot_predictor(
            config=config,
            train_tsdf=train_tsdf,
            horizon=horizon,
            path=str(resolve_path(config["paths"]["models_dir"]) / f"sensitivity_zeroshot_H{horizon}"),
            logger=logger,
        )
        zeroshot_forecast = rolling_forecast(
            zeroshot_predictor, full_df, eval_start_idx, horizon, stride, max_context,
            config=config, origins=origins, logger=logger,
        )
        assert zeroshot_forecast["origins"] == list(origins)
        save_forecast(zeroshot_forecast, SCHEME_ZEROSHOT, horizon, "test", config, logger)
        forecasts[SCHEME_ZEROSHOT] = zeroshot_forecast

        finetune_info: dict[str, Any] | None = None
        if include_finetune:
            val_tsdf = to_model_tsdf(
                frames["val"], config, end_position=len(frames["train"]) + len(frames["val"])
            )
            if reuse_finetune_params:
                best_params = reused_best_params
                finetune_info = {
                    "best_params": best_params,
                    "retuned": False,
                    "source": f"H={reference_horizon} (hasil utama, tidak di-tuning ulang)",
                }
            else:
                tuning = tune_finetune_hyperparams(
                    train_tsdf, val_tsdf, config=config, horizon=horizon, logger=logger
                )
                best_params = tuning["best_params"]
                finetune_info = {
                    "best_params": best_params,
                    "best_val_wql": tuning["best_score"],
                    "retuned": True,
                }

            finetuned_predictor = build_finetuned_predictor(
                train_tsdf,
                best_params,
                config=config,
                horizon=horizon,
                path=str(resolve_path(config["paths"]["models_dir"]) / f"sensitivity_finetuned_H{horizon}"),
                logger=logger,
            )
            finetuned_forecast = rolling_forecast(
                finetuned_predictor, full_df, eval_start_idx, horizon, stride, max_context,
                config=config, origins=origins, logger=logger,
            )
            assert finetuned_forecast["origins"] == list(origins)
            save_forecast(finetuned_forecast, SCHEME_FINETUNED, horizon, "test", config, logger)
            forecasts[SCHEME_FINETUNED] = finetuned_forecast

        naive_point = forecasts[SCHEME_NAIVE]["y_pred"]
        metrics = {
            scheme: aggregate_metrics(
                forecast["y_true"], forecast["y_pred_quantiles"], quantile_levels,
                y_naive=naive_point, config=config,
            )
            for scheme, forecast in forecasts.items()
        }

        dm_pairs = [
            (SCHEME_ZEROSHOT, SCHEME_FINETUNED, True),
            (SCHEME_ZEROSHOT, SCHEME_NAIVE, False),
            (SCHEME_FINETUNED, SCHEME_NAIVE, False),
        ]
        dm_tests = _run_dm_pairs(forecasts, dm_pairs, horizon, alpha, SCHEME_LABELS)

        results_per_horizon[str(horizon)] = {
            "horizon": int(horizon),
            "n_windows": len(origins),
            "test_start": str(pd.Timestamp(forecasts[SCHEME_NAIVE]["origin_dates"][0]).date()),
            "test_end": str(pd.Timestamp(forecasts[SCHEME_NAIVE]["origin_dates"][-1]).date()),
            "schemes_evaluated": list(forecasts),
            "finetune_selection": finetune_info,
            "metrics": metrics,
            "diebold_mariano": dm_tests,
            "source": "dihitung pada analisis sensitivitas ini",
        }

        overall_line = "  ".join(
            f"{SCHEME_LABELS[scheme]}: MAE={metrics[scheme]['overall']['mae']:.4f}"
            for scheme in forecasts
        )
        logger.info("Horizon H=%d selesai (%d jendela) -- %s", horizon, len(origins), overall_line)

    # --- Sertakan H utama sebagai baris pembanding, dimuat apa adanya ---
    final_results_path = resolve_path(config["paths"]["metrics_dir"]) / "final_results.json"
    if final_results_path.is_file():
        with final_results_path.open("r", encoding="utf-8") as file:
            main_final_results = json.load(file)
        results_per_horizon[str(reference_horizon)] = {
            "horizon": reference_horizon,
            "n_windows": main_final_results["protocol"]["n_windows"],
            "test_start": main_final_results["protocol"]["test_start"],
            "test_end": main_final_results["protocol"]["test_end"],
            "schemes_evaluated": main_final_results["protocol"]["schemes_evaluated"],
            "finetune_selection": main_final_results.get("finetune_selection"),
            "metrics": main_final_results["metrics"],
            "diebold_mariano": main_final_results["diebold_mariano"],
            "source": "results/metrics/final_results.json (hasil utama, dimuat apa adanya, TIDAK dihitung ulang)",
        }
    else:
        logger.warning(
            "final_results.json tidak ditemukan -- H utama (%d) tidak disertakan sebagai pembanding.",
            reference_horizon,
        )

    output = {
        "protocol": {
            "horizons_sensitivity_requested": [int(h) for h in horizons],
            "reference_horizon_main": reference_horizon,
            "stride": int(stride),
            "include_finetune": include_finetune,
            "reuse_finetune_params": reuse_finetune_params,
            "note": (
                "Disimpan terpisah dari results/metrics/final_results.json (hasil utama) "
                "agar tidak tercampur. Tiap horizon memakai daftar origin "
                "sendiri -- jumlah jendela berbeda dari H utama, sehingga yang "
                "dibandingkan lintas horizon adalah metrik teragregasi, bukan jendela "
                "per jendela."
            ),
        },
        "results": results_per_horizon,
    }

    output_path = save_json(
        output, resolve_path(config["paths"]["metrics_dir"]) / "sensitivity_horizon.json"
    )
    logger.info("Hasil sensitivitas horizon tersimpan di %s", output_path)
    return output


# =============================================================================
# 2. Ablasi kovariat
# =============================================================================


def run_covariate_ablation(
    horizon: int | None = None,
    base_scheme: str | None = None,
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Membandingkan tiga kondisi kovariat input pada skema Chronos-2 terbaik.

    Kondisi (a) hanya Perak, (b) Perak+Emas, (c) Perak+Emas+DXY dijalankan pada
    horizon dan daftar origin **yang sama dengan hasil utama** (dijamin lewat
    ``assert`` terhadap ``{base_scheme}_H{horizon}.parquet``), sehingga
    perbandingannya adil. Kondisi (c) memakai seluruh kovariat pada config --
    identik dengan skema utama -- sehingga hasilnya dimuat ulang dari berkas
    hasil utama, bukan dijalankan ulang.

    Ablasi ini hanya diimplementasikan untuk skema **zero-shot** (tanpa
    pelatihan): mengulanginya untuk fine-tuned berarti tuning grid penuh
    dikalikan jumlah kondisi, yang tidak layak secara komputasi pada perangkat
    mana pun yang dipakai penelitian ini (lihat docstring modul). Bila skema terbaik hasil utama ternyata fine-tuned,
    fungsi ini berhenti dengan pesan yang jelas alih-alih diam-diam memakai
    skema yang salah.

    Args:
        horizon: Panjang horizon H. Bila ``None``, diambil dari
            ``model.prediction_length`` (H utama).
        base_scheme: Skema dasar ablasi. Bila ``None``, ditentukan otomatis
            lewat :func:`_determine_best_scheme`.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dictionary hasil ablasi, juga disimpan ke
        ``results/metrics/sensitivity_covariate_ablation.json``.

    Raises:
        SensitivityError: Bila skema terbaik bukan zero-shot, atau berkas
            hasil utama untuk skema/horizon tersebut belum ada.
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("sensitivity", log_dir=config.get("paths", {}).get("logs_dir", "logs"))
    if horizon is None:
        horizon = config["model"]["prediction_length"]
    if base_scheme is None:
        base_scheme = _determine_best_scheme(config, logger)

    if base_scheme != SCHEME_ZEROSHOT:
        raise SensitivityError(
            f"Ablasi kovariat hanya diimplementasikan untuk skema '{SCHEME_ZEROSHOT}' "
            f"(tanpa fine-tuning, sehingga biayanya rendah dikali 2 kondisi baru). "
            f"Skema terbaik hasil utama terdeteksi '{base_scheme}', yang memerlukan "
            f"tuning grid penuh per kondisi -- belum diimplementasikan karena "
            f"biayanya. Panggil dengan base_scheme='{SCHEME_ZEROSHOT}' secara "
            f"eksplisit bila tetap ingin melanjutkan dengan skema itu."
        )

    logger.info("=" * 78)
    logger.info(
        "ABLASI KOVARIAT -- skema dasar: %s, H=%d", SCHEME_LABELS[base_scheme], horizon
    )
    logger.info("=" * 78)

    covariates_all = get_covariate_columns(config)
    if len(covariates_all) < 2:
        raise SensitivityError(
            f"Ablasi kovariat mengasumsikan tepat 2 kovariat (Emas, DXY) pada config, "
            f"diterima {len(covariates_all)}: {covariates_all}."
        )
    conditions: dict[str, list[str]] = {
        "silver_only": [],
        "silver_gold": covariates_all[:1],
        "silver_gold_dxy": covariates_all[:2],
    }
    logger.info("Kondisi kovariat: %s", conditions)

    frames = load_processed_frames(("train", "val", "test"), config=config, logger=logger)
    full_df = pd.concat([frames["train"], frames["val"], frames["test"]])
    full_df.index.name = DATE_INDEX_NAME
    eval_start_idx = len(frames["train"]) + len(frames["val"])
    stride = config["evaluation"]["stride"]
    max_context = config["model"]["max_context"]
    quantile_levels = config["model"]["quantile_levels"]

    origins = build_origins(len(full_df), eval_start_idx, horizon, stride)

    main_forecast = load_forecast_arrays(base_scheme, horizon, "test", config)
    if main_forecast["origins"] != list(origins):
        raise SensitivityError(
            "Daftar origin ablasi kovariat tidak sama dengan hasil utama -- "
            "protokol rolling origin dilanggar. Periksa apakah config berubah sejak hasil "
            "utama dihasilkan."
        )
    naive_point = load_forecast_arrays(SCHEME_NAIVE, horizon, "test", config)["y_pred"]
    logger.info(
        "Origin diverifikasi identik dengan hasil utama (%s_H%d.parquet, %d jendela).",
        base_scheme, horizon, len(origins),
    )

    forecasts: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}

    for condition_name, condition_covariates in conditions.items():
        logger.info("-" * 78)
        logger.info(
            "Kondisi '%s' (%s): kovariat=%s",
            condition_name, CONDITION_LABELS[condition_name], condition_covariates or "(tidak ada)",
        )

        if condition_covariates == covariates_all:
            # Identik dengan skema utama -- dimuat ulang, tidak dijalankan lagi.
            forecast = main_forecast
            logger.info("Dimuat ulang dari %s_H%d.parquet (identik dengan skema utama).", base_scheme, horizon)
        else:
            train_tsdf = to_model_tsdf(frames["train"], config, covariate_columns=condition_covariates)
            predictor = build_zeroshot_predictor(
                config=config,
                train_tsdf=train_tsdf,
                horizon=horizon,
                path=str(
                    resolve_path(config["paths"]["models_dir"])
                    / f"sensitivity_ablation_{condition_name}_H{horizon}"
                ),
                logger=logger,
            )
            forecast = rolling_forecast(
                predictor, full_df, eval_start_idx, horizon, stride, max_context,
                config=config, origins=origins, logger=logger,
                covariate_columns=condition_covariates,
            )
            assert forecast["origins"] == list(origins)
            save_forecast(forecast, f"ablation_{condition_name}", horizon, "test", config, logger)

        forecasts[condition_name] = forecast
        metrics[condition_name] = aggregate_metrics(
            forecast["y_true"], forecast["y_pred_quantiles"], quantile_levels,
            y_naive=naive_point, config=config,
        )
        overall = metrics[condition_name]["overall"]
        logger.info(
            "Kondisi '%s': MAE=%.4f CRPS=%.4f MASE=%s",
            condition_name, overall["mae"], overall["crps"], _format_number(overall.get("mase")),
        )

    alpha = config["evaluation"]["diebold_mariano"]["alpha"]
    dm_pairs = [
        ("silver_only", "silver_gold_dxy", True),
        ("silver_gold", "silver_gold_dxy", False),
        ("silver_only", "silver_gold", False),
    ]
    dm_tests = _run_dm_pairs(
        forecasts, dm_pairs, horizon, alpha, CONDITION_LABELS,
        key_a="condition_a", key_b="condition_b", better_key="better_condition",
    )

    mi_crossref = _load_mutual_information_crossref(config, logger)

    result = {
        "label": f"Ablasi kovariat pada skema terbaik hasil utama ({SCHEME_LABELS[base_scheme]})",
        "protocol": {
            "base_scheme": base_scheme,
            "horizon": int(horizon),
            "stride": int(stride),
            "n_windows": len(origins),
            "test_start": str(pd.Timestamp(main_forecast["origin_dates"][0]).date()),
            "test_end": str(pd.Timestamp(main_forecast["origin_dates"][-1]).date()),
            "conditions": conditions,
            "condition_labels": CONDITION_LABELS,
            "note": (
                "Kondisi 'silver_gold_dxy' identik dengan skema utama pada horizon ini "
                "sehingga dimuat ulang dari hasil utama, bukan dijalankan ulang."
            ),
        },
        "metrics": metrics,
        "diebold_mariano": dm_tests,
        "mutual_information_crossref": mi_crossref,
    }

    output_path = save_json(
        result, resolve_path(config["paths"]["metrics_dir"]) / "sensitivity_covariate_ablation.json"
    )
    logger.info("Hasil ablasi kovariat tersimpan di %s", output_path)
    return result


# =============================================================================
# 3. Skenario oracle / ex-post (opsional)
# =============================================================================


def run_oracle_ablation(
    horizon: int | None = None,
    base_scheme: str | None = None,
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Skenario ORACLE / EX-POST -- kovariat sebagai known-future (opsional).

    **TIDAK REALISTIS.** Nilai AKTUAL Emas dan Dollar Index pada H langkah ke
    depan diberikan ke model lewat argumen ``known_covariates``. Ablasi ini sah
    dilakukan sebagai *upper bound* teoretis, dengan syarat WAJIB dilabeli. Hasilnya menjawab: seberapa besar
    ruang perbaikan yang tersisa bila model punya akses sempurna ke masa depan
    kovariat -- bukan estimasi performa yang dapat dicapai saat peramalan
    sungguhan.

    Args:
        horizon: Panjang horizon H. Bila ``None``, diambil dari
            ``model.prediction_length`` (H utama).
        base_scheme: Skema realistis pembanding. Bila ``None``, ditentukan
            otomatis lewat :func:`_determine_best_scheme`.
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dictionary hasil, juga disimpan ke
        ``results/metrics/sensitivity_oracle.json``. Field ``label`` pada level
        teratas berisi label ex-post yang WAJIB dikutip ulang setiap kali
        angkanya dilaporkan.

    Raises:
        SensitivityError: Bila skema dasar bukan zero-shot (lihat
            :func:`run_covariate_ablation` untuk alasan yang sama).
    """
    if config is None:
        config = load_config()
    if logger is None:
        logger = setup_logger("sensitivity", log_dir=config.get("paths", {}).get("logs_dir", "logs"))
    if horizon is None:
        horizon = config["model"]["prediction_length"]
    if base_scheme is None:
        base_scheme = _determine_best_scheme(config, logger)

    if base_scheme != SCHEME_ZEROSHOT:
        raise SensitivityError(
            f"Skenario oracle/ex-post hanya diimplementasikan untuk skema "
            f"'{SCHEME_ZEROSHOT}' (tanpa fine-tuning). Skema terbaik hasil utama "
            f"terdeteksi '{base_scheme}'. Panggil dengan base_scheme='{SCHEME_ZEROSHOT}' "
            f"secara eksplisit bila tetap ingin melanjutkan dengan skema itu."
        )

    logger.warning("=" * 78)
    logger.warning("SKENARIO ORACLE / EX-POST -- TIDAK REALISTIS (batas atas teoretis)")
    logger.warning("=" * 78)

    covariates_all = get_covariate_columns(config)

    frames = load_processed_frames(("train", "val", "test"), config=config, logger=logger)
    full_df = pd.concat([frames["train"], frames["val"], frames["test"]])
    full_df.index.name = DATE_INDEX_NAME
    eval_start_idx = len(frames["train"]) + len(frames["val"])
    stride = config["evaluation"]["stride"]
    max_context = config["model"]["max_context"]
    quantile_levels = config["model"]["quantile_levels"]

    origins = build_origins(len(full_df), eval_start_idx, horizon, stride)

    main_forecast = load_forecast_arrays(base_scheme, horizon, "test", config)
    if main_forecast["origins"] != list(origins):
        raise SensitivityError(
            "Daftar origin skenario oracle tidak sama dengan hasil utama -- "
            "protokol rolling origin dilanggar."
        )
    naive_point = load_forecast_arrays(SCHEME_NAIVE, horizon, "test", config)["y_pred"]

    train_tsdf = to_model_tsdf(frames["train"], config)
    predictor = build_oracle_predictor(
        config=config,
        train_tsdf=train_tsdf,
        horizon=horizon,
        path=str(resolve_path(config["paths"]["models_dir"]) / f"sensitivity_oracle_H{horizon}"),
        logger=logger,
        known_covariates_columns=covariates_all,
    )
    oracle_forecast = rolling_forecast(
        predictor, full_df, eval_start_idx, horizon, stride, max_context,
        config=config, origins=origins, logger=logger,
        known_covariates_columns=covariates_all,
    )
    assert oracle_forecast["origins"] == list(origins)

    oracle_scheme_name = "oracle_expost"
    save_forecast(oracle_forecast, oracle_scheme_name, horizon, "test", config, logger)

    metrics_realistic = aggregate_metrics(
        main_forecast["y_true"], main_forecast["y_pred_quantiles"], quantile_levels,
        y_naive=naive_point, config=config,
    )
    metrics_oracle = aggregate_metrics(
        oracle_forecast["y_true"], oracle_forecast["y_pred_quantiles"], quantile_levels,
        y_naive=naive_point, config=config,
    )

    alpha = config["evaluation"]["diebold_mariano"]["alpha"]
    forecasts_for_dm = {base_scheme: main_forecast, oracle_scheme_name: oracle_forecast}
    dm_labels = {base_scheme: f"{SCHEME_LABELS[base_scheme]} (realistis)", oracle_scheme_name: "ORACLE/EX-POST (TIDAK REALISTIS)"}
    dm_tests = _run_dm_pairs(
        forecasts_for_dm, [(base_scheme, oracle_scheme_name, True)], horizon, alpha, dm_labels
    )

    mae_realistic = metrics_realistic["overall"]["mae"]
    mae_oracle = metrics_oracle["overall"]["mae"]
    crps_realistic = metrics_realistic["overall"]["crps"]
    crps_oracle = metrics_oracle["overall"]["crps"]

    result = {
        "label": ORACLE_LABEL,
        "protocol": {
            "base_scheme_realistic": base_scheme,
            "oracle_scheme": oracle_scheme_name,
            "horizon": int(horizon),
            "n_windows": len(origins),
            "test_start": str(pd.Timestamp(main_forecast["origin_dates"][0]).date()),
            "test_end": str(pd.Timestamp(main_forecast["origin_dates"][-1]).date()),
            "known_covariates_columns": list(covariates_all),
        },
        "metrics": {base_scheme: metrics_realistic, oracle_scheme_name: metrics_oracle},
        "diebold_mariano": dm_tests,
        "gap_summary_ex_post": {
            "mae_gap": mae_realistic - mae_oracle,
            "mae_gap_pct": (mae_realistic - mae_oracle) / mae_realistic * 100.0,
            "crps_gap": crps_realistic - crps_oracle,
            "crps_gap_pct": (crps_realistic - crps_oracle) / crps_realistic * 100.0,
        },
    }

    output_path = save_json(
        result, resolve_path(config["paths"]["metrics_dir"]) / "sensitivity_oracle.json"
    )
    logger.warning("Hasil skenario ORACLE/EX-POST (TIDAK REALISTIS) tersimpan di %s", output_path)
    return result


# =============================================================================
# Ringkasan: markdown + figure
# =============================================================================


def build_summary_markdown(
    horizon_result: dict[str, Any] | None,
    ablation_result: dict[str, Any] | None,
    oracle_result: dict[str, Any] | None,
    config: dict[str, Any],
) -> str:
    """Menyusun ``results/metrics/sensitivity_summary.md``.

    Args:
        horizon_result: Keluaran :func:`run_horizon_sensitivity`, atau ``None``
            bila belum dijalankan.
        ablation_result: Keluaran :func:`run_covariate_ablation`, atau ``None``.
        oracle_result: Keluaran :func:`run_oracle_ablation`, atau ``None``
            (skenario ini memang opsional).
        config: Konfigurasi project.

    Returns:
        Isi berkas markdown sebagai string.
    """
    alpha = config["evaluation"]["diebold_mariano"]["alpha"]
    lines: list[str] = [
        "# Analisis Sensitivitas",
        "",
        "Disimpan **terpisah** dari hasil utama "
        "(`results/metrics/final_results.json`, H = 10) agar tidak tercampur. "
        "Sumber angka: `sensitivity_horizon.json`, `sensitivity_covariate_ablation.json`, "
        "`sensitivity_oracle.json`, dan gambar `results/figures/sensitivity_summary.png`.",
        "",
    ]

    # --- 1. Sensitivitas horizon ---
    lines += ["## 1. Sensitivitas horizon", ""]
    if horizon_result is None:
        lines += ["_Belum dijalankan._", ""]
    else:
        protocol = horizon_result["protocol"]
        lines += [
            f"Protokol rolling origin diulang pada H = "
            f"{protocol['horizons_sensitivity_requested']}, dibandingkan dengan H utama "
            f"= {protocol['reference_horizon_main']}. Tiap horizon memakai daftar origin "
            "sendiri (jumlah jendela berbeda), sehingga yang dibandingkan adalah metrik "
            "teragregasi, bukan jendela per jendela.",
            "",
        ]
        if not protocol["include_finetune"]:
            lines.append(
                "> Skema **Fine-Tuned tidak disertakan** pada horizon sensitivitas ini "
                "(`include_finetune=False`) karena biaya tuning grid penuh harus "
                "diulang untuk tiap horizon."
            )
            lines.append("")
        elif protocol["reuse_finetune_params"]:
            lines.append(
                "> Skema Fine-Tuned pada H selain H utama memakai **ulang** "
                "hyperparameter terbaik hasil tuning H utama, **TANPA tuning grid "
                "ulang per horizon** (penyimpangan eksplisit dari protokol pemilihan "
                "hyperparameter, demi kelayakan komputasi: tuning ulang berarti "
                "mengulang seluruh pencarian 18 kandidat untuk tiap horizon "
                "-- lihat catatan pada `sensitivity_horizon.json`)."
            )
            lines.append("")

        lines += [
            "| H | Skema | n jendela | MAE | RMSE | MASE | CRPS | nCRPS | Cov80 | Cov95 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for horizon_key in sorted(horizon_result["results"], key=lambda k: int(k)):
            entry = horizon_result["results"][horizon_key]
            marker = " **(H utama)**" if int(horizon_key) == protocol["reference_horizon_main"] else ""
            for scheme in (SCHEME_NAIVE, SCHEME_ZEROSHOT, SCHEME_FINETUNED):
                if scheme not in entry["metrics"]:
                    continue
                overall = entry["metrics"][scheme]["overall"]
                lines.append(
                    "| {h}{marker} | {scheme} | {n} | {mae} | {rmse} | {mase} | {crps} | "
                    "{ncrps} | {cov80} | {cov95} |".format(
                        h=horizon_key,
                        marker=marker,
                        scheme=SCHEME_LABELS.get(scheme, scheme),
                        n=entry["n_windows"],
                        mae=_format_number(overall["mae"]),
                        rmse=_format_number(overall["rmse"]),
                        mase=_format_number(overall.get("mase")),
                        crps=_format_number(overall["crps"]),
                        ncrps=_format_number(overall["normalized_crps"], 6),
                        cov80=_format_number(overall["coverage"].get("0.8"), 4),
                        cov95=_format_number(overall["coverage"].get("0.95"), 4),
                    )
                )
        lines.append("")

        lines += ["**Uji Diebold-Mariano per horizon** (rugi = galat absolut, HAC Newey-West):", ""]
        lines += ["| H | Perbandingan | p-value | Signifikan (α={:g}) |".format(alpha), "|---|---|---|---|"]
        for horizon_key in sorted(horizon_result["results"], key=lambda k: int(k)):
            entry = horizon_result["results"][horizon_key]
            for test in entry["diebold_mariano"]:
                if not test["primary"]:
                    continue
                lines.append(
                    "| {h} | {label} | {p} | {sig} |".format(
                        h=horizon_key,
                        label=test["label"],
                        p=_format_number(test["p_value"], 4),
                        sig="ya" if test["significant"] else "tidak",
                    )
                )
        lines.append("")

    # --- 2. Ablasi kovariat ---
    lines += ["## 2. Ablasi kovariat (skema terbaik)", ""]
    if ablation_result is None:
        lines += ["_Belum dijalankan._", ""]
    else:
        protocol = ablation_result["protocol"]
        lines += [
            f"Skema dasar: **{SCHEME_LABELS.get(protocol['base_scheme'], protocol['base_scheme'])}** "
            f"(terpilih otomatis: nCRPS terkecil pada hasil utama). H = {protocol['horizon']}, "
            f"{protocol['n_windows']} jendela ({protocol['test_start']} s.d. {protocol['test_end']}) "
            "-- daftar origin diverifikasi identik dengan hasil utama.",
            "",
            "| Kondisi | Kovariat | MAE | CRPS | MASE | nCRPS |",
            "|---|---|---|---|---|---|",
        ]
        for condition_name, overall_metrics in ablation_result["metrics"].items():
            overall = overall_metrics["overall"]
            covariates = protocol["conditions"][condition_name]
            lines.append(
                "| {label} | {cov} | {mae} | {crps} | {mase} | {ncrps} |".format(
                    label=protocol["condition_labels"][condition_name],
                    cov=", ".join(covariates) if covariates else "(tidak ada)",
                    mae=_format_number(overall["mae"]),
                    crps=_format_number(overall["crps"]),
                    mase=_format_number(overall.get("mase")),
                    ncrps=_format_number(overall["normalized_crps"], 6),
                )
            )
        lines.append("")

        lines += ["**Uji Diebold-Mariano antar kondisi** (menguji apakah kovariat benar-benar membantu):", ""]
        lines += ["| Perbandingan | Selisih rugi rata-rata | p-value | Signifikan (α={:g}) | Lebih baik |".format(alpha), "|---|---|---|---|---|"]
        for test in ablation_result["diebold_mariano"]:
            marker = " **(uji utama)**" if test["primary"] else ""
            better = (
                "seri (tidak berbeda)" if test["better_condition"] is None
                else CONDITION_LABELS.get(test["better_condition"], test["better_condition"])
            )
            lines.append(
                "| {label}{marker} | {diff} | {p} | {sig} | {better} |".format(
                    label=test["label"], marker=marker,
                    diff=_format_number(test["mean_loss_differential"], 6),
                    p=_format_number(test["p_value"], 4),
                    sig="ya" if test["significant"] else "tidak",
                    better=better if test["significant"] else "tidak terbukti berbeda",
                )
            )
        lines.append("")

        primary_test = next((t for t in ablation_result["diebold_mariano"] if t["primary"]), None)
        if primary_test is not None:
            if primary_test["significant"]:
                verdict = (
                    f"Menambahkan kovariat (Emas + DXY) **terbukti signifikan** mengubah "
                    f"akurasi dibanding hanya Perak (p = {primary_test['p_value']:.4f} < {alpha:g})."
                )
            else:
                verdict = (
                    f"Menambahkan kovariat (Emas + DXY) **tidak terbukti signifikan** "
                    f"mengubah akurasi dibanding hanya Perak (p = {primary_test['p_value']:.4f} "
                    f">= {alpha:g}); secara empiris kovariat ini tidak terbukti membantu "
                    "pada protokol evaluasi ini."
                )
            lines += [f"**Kesimpulan uji utama:** {verdict}", ""]

        mi_crossref = ablation_result.get("mutual_information_crossref")
        if mi_crossref:
            lines += [
                "**Sambungan dengan Mutual Information:** "
                f"(sumber: `{mi_crossref['source']}`)",
                "",
            ]
            for mode, pairs in mi_crossref["variants"].items():
                pretty_pairs = ", ".join(f"{cov}={mi:.4f}" for cov, mi in pairs.items())
                lines.append(f"- MI ({mode}): {pretty_pairs}")
            lines += [
                "",
                "Bila MI (terutama pada log-return, varian yang secara statistik sah -- "
                "lihat Notebook 02) kecil untuk kedua kovariat, hasil ini KONSISTEN "
                "dengan uji Diebold-Mariano di atas: kovariat secara empiris tidak "
                "banyak menambah informasi bagi ramalan Perak.",
                "",
            ]

    # --- 3. Skenario oracle / ex-post ---
    lines += [
        "## 3. Skenario ORACLE / EX-POST — TIDAK REALISTIS (opsional, batas atas teoretis)",
        "",
    ]
    if oracle_result is None:
        lines += [
            "_Belum dijalankan (opsional)._",
            "",
        ]
    else:
        lines += [
            f"> **{oracle_result['label']}**",
            "",
        ]
        protocol = oracle_result["protocol"]
        lines += [
            f"H = {protocol['horizon']}, {protocol['n_windows']} jendela "
            f"({protocol['test_start']} s.d. {protocol['test_end']}), kovariat known-future: "
            f"{', '.join(protocol['known_covariates_columns'])}.",
            "",
            "| Skema | MAE | CRPS | Cov80 | Cov95 |",
            "|---|---|---|---|---|",
        ]
        for scheme_name, overall_metrics in oracle_result["metrics"].items():
            overall = overall_metrics["overall"]
            label = (
                f"{SCHEME_LABELS.get(scheme_name, scheme_name)} (realistis)"
                if scheme_name == protocol["base_scheme_realistic"]
                else "**ORACLE/EX-POST (TIDAK REALISTIS)**"
            )
            lines.append(
                "| {label} | {mae} | {crps} | {cov80} | {cov95} |".format(
                    label=label,
                    mae=_format_number(overall["mae"]),
                    crps=_format_number(overall["crps"]),
                    cov80=_format_number(overall["coverage"].get("0.8"), 4),
                    cov95=_format_number(overall["coverage"].get("0.95"), 4),
                )
            )
        gap = oracle_result["gap_summary_ex_post"]
        lines += [
            "",
            f"Selisih MAE (realistis − oracle): {gap['mae_gap']:.4f} USD "
            f"({gap['mae_gap_pct']:.2f}% dari MAE realistis) -- **ini bukan potensi "
            "perbaikan yang dapat dicapai model manapun secara realistis**, melainkan "
            "ukuran seberapa besar informasi yang (secara ilegitimate) diberikan kepada "
            "model pada skenario ex-post ini.",
            "",
        ]
        primary = next(iter(oracle_result["diebold_mariano"]), None)
        if primary is not None:
            lines += [
                f"Uji Diebold-Mariano (realistis vs oracle/ex-post): p = "
                f"{primary['p_value']:.4f} "
                f"({'signifikan' if primary['significant'] else 'tidak signifikan'} pada "
                f"α = {alpha:g}).",
                "",
            ]

    lines += [
        "---",
        "",
        "Gambar ringkasan ketiga analisis: `results/figures/sensitivity_summary.png`.",
        "",
    ]

    return "\n".join(lines)


def plot_sensitivity_summary(
    horizon_result: dict[str, Any] | None,
    ablation_result: dict[str, Any] | None,
    oracle_result: dict[str, Any] | None,
    output_path: str | Path,
    dpi: int,
    config: dict[str, Any],
) -> Path:
    """Menggambar satu figure ringkasan tiga panel untuk ketiga analisis sensitivitas.

    Panel kiri: MAE terhadap horizon per skema. Panel tengah: MAE per kondisi
    ablasi kovariat (skema terbaik), dengan garis referensi naive. Panel kanan:
    skema realistis vs skenario oracle/ex-post, diberi tanda visual tegas
    (bilah bercorak, berlabel merah) agar tidak mungkin disalahartikan sebagai
    hasil yang dapat dicapai secara nyata.

    Args:
        horizon_result: Keluaran :func:`run_horizon_sensitivity`, atau ``None``.
        ablation_result: Keluaran :func:`run_covariate_ablation`, atau ``None``.
        oracle_result: Keluaran :func:`run_oracle_ablation`, atau ``None``.
        output_path: Lokasi berkas gambar keluaran.
        dpi: Resolusi gambar.
        config: Konfigurasi project.

    Returns:
        Path absolut berkas gambar yang tersimpan.
    """
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), facecolor=_SURFACE)

    for ax in axes:
        ax.set_facecolor(_SURFACE)
        ax.grid(axis="y", color=_GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_BASELINE)
        ax.tick_params(colors=_INK_MUTED, labelsize=9, length=0)

    # --- Panel A: MAE vs horizon ---
    ax = axes[0]
    if horizon_result is None:
        ax.text(0.5, 0.5, "Belum dijalankan", ha="center", va="center", color=_INK_MUTED)
    else:
        results = horizon_result["results"]
        horizons_sorted = sorted(results, key=lambda k: int(k))
        for scheme in (SCHEME_NAIVE, SCHEME_ZEROSHOT, SCHEME_FINETUNED):
            xs, ys = [], []
            for h_key in horizons_sorted:
                entry = results[h_key]
                if scheme in entry["metrics"]:
                    xs.append(int(h_key))
                    ys.append(entry["metrics"][scheme]["overall"]["mae"])
            if xs:
                ax.plot(
                    xs, ys, color=SCHEME_COLORS[scheme], marker="o", markersize=5,
                    markeredgecolor=_SURFACE, markeredgewidth=0.8, linewidth=1.8,
                    label=SCHEME_LABELS[scheme],
                )
        ax.set_xticks([int(h) for h in horizons_sorted])
        ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=_INK_SECONDARY)
    ax.set_xlabel("Horizon H (hari perdagangan)", fontsize=9.5, color=_INK_SECONDARY)
    ax.set_ylabel("MAE (USD)", fontsize=9.5, color=_INK_SECONDARY)
    ax.set_title("1. Sensitivitas horizon", fontsize=11, color=_INK_PRIMARY, pad=10)

    # --- Panel B: ablasi kovariat ---
    ax = axes[1]
    if ablation_result is None:
        ax.text(0.5, 0.5, "Belum dijalankan", ha="center", va="center", color=_INK_MUTED)
    else:
        condition_order = ["silver_only", "silver_gold", "silver_gold_dxy"]
        condition_order = [c for c in condition_order if c in ablation_result["metrics"]]
        maes = [ablation_result["metrics"][c]["overall"]["mae"] for c in condition_order]
        x_positions = np.arange(len(condition_order))
        ax.bar(
            x_positions, maes, color=SCHEME_COLORS[SCHEME_ZEROSHOT], alpha=0.75,
            width=0.55, zorder=3,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [CONDITION_LABELS[c].replace(" + ", "+\n") for c in condition_order], fontsize=8
        )
        for x, mae in zip(x_positions, maes):
            ax.text(x, mae, f"{mae:.3f}", ha="center", va="bottom", fontsize=8, color=_INK_SECONDARY)
    ax.set_ylabel("MAE (USD)", fontsize=9.5, color=_INK_SECONDARY)
    base_label = SCHEME_LABELS.get(
        ablation_result["protocol"]["base_scheme"], ""
    ) if ablation_result else ""
    ax.set_title(
        f"2. Ablasi kovariat ({base_label})" if base_label else "2. Ablasi kovariat",
        fontsize=11, color=_INK_PRIMARY, pad=10,
    )

    # --- Panel C: oracle / ex-post ---
    ax = axes[2]
    if oracle_result is None:
        ax.text(0.5, 0.5, "Belum dijalankan (opsional)", ha="center", va="center", color=_INK_MUTED)
    else:
        protocol = oracle_result["protocol"]
        realistic_scheme = protocol["base_scheme_realistic"]
        oracle_scheme = protocol["oracle_scheme"]
        mae_realistic = oracle_result["metrics"][realistic_scheme]["overall"]["mae"]
        mae_oracle = oracle_result["metrics"][oracle_scheme]["overall"]["mae"]

        ax.bar(
            [0], [mae_realistic], color=SCHEME_COLORS[SCHEME_ZEROSHOT], alpha=0.75,
            width=0.55, zorder=3, label=f"{SCHEME_LABELS.get(realistic_scheme, realistic_scheme)} (realistis)",
        )
        ax.bar(
            [1], [mae_oracle], color=_ORACLE_COLOR, alpha=0.55, hatch="///",
            edgecolor=_ORACLE_COLOR, width=0.55, zorder=3, label="ORACLE/EX-POST",
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Realistis\n(past-only)", "EX-POST\n(TIDAK REALISTIS)"], fontsize=8.5)
        for x, mae in zip([0, 1], [mae_realistic, mae_oracle]):
            ax.text(x, mae, f"{mae:.3f}", ha="center", va="bottom", fontsize=8, color=_INK_SECONDARY)
        ax.text(
            0.5, 1.14, "BUKAN skenario realistis -- batas atas teoretis",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color=_ORACLE_COLOR,
            fontweight="bold",
        )
    ax.set_ylabel("MAE (USD)", fontsize=9.5, color=_INK_SECONDARY)
    ax.set_title("3. Realistis vs ORACLE/EX-POST", fontsize=11, color=_INK_PRIMARY, pad=10)

    figure.suptitle(
        "Ringkasan analisis sensitivitas: horizon, ablasi kovariat, skenario oracle/ex-post",
        fontsize=12.5, color=_INK_PRIMARY, y=1.06,
    )
    figure.tight_layout()

    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


# =============================================================================
# Titik masuk mandiri
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Membaca argumen baris perintah untuk eksekusi mandiri.

    Args:
        argv: Daftar argumen. Bila ``None``, diambil dari ``sys.argv``.

    Returns:
        Namespace argumen.
    """
    parser = argparse.ArgumentParser(
        description="Analisis sensitivitas: horizon, ablasi kovariat, skenario oracle/ex-post."
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=None, help="Daftar horizon (bawaan: config).")
    parser.add_argument("--skip-horizon", action="store_true", help="Lewati sensitivitas horizon.")
    parser.add_argument("--skip-covariate", action="store_true", help="Lewati ablasi kovariat.")
    parser.add_argument("--skip-oracle", action="store_true", help="Lewati skenario oracle/ex-post.")
    parser.add_argument(
        "--skip-finetune-horizon", action="store_true",
        help="Sensitivitas horizon: lewati skema fine-tuned (naive+zero-shot saja).",
    )
    parser.add_argument(
        "--reuse-finetune-params", action="store_true",
        help="Sensitivitas horizon: pakai ulang hyperparameter terbaik H utama, jangan tuning ulang.",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Jangan jalankan analisis apa pun; susun ulang sensitivity_summary.md + figure dari berkas JSON yang sudah ada.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Titik masuk eksekusi mandiri modul.

    Args:
        argv: Daftar argumen baris perintah.

    Returns:
        Dictionary ``{"horizon", "covariate_ablation", "oracle", "summary_path", "figure_path"}``.
    """
    args = parse_args(argv)
    config = load_config()
    logger = setup_logger("sensitivity", log_dir=config.get("paths", {}).get("logs_dir", "logs"))
    metrics_dir = resolve_path(config["paths"]["metrics_dir"])

    if args.summary_only:
        logger.info("--summary-only: memuat ulang JSON yang ada, tidak menjalankan analisis apa pun.")
        horizon_result = _load_json_if_exists(metrics_dir / "sensitivity_horizon.json")
        ablation_result = _load_json_if_exists(metrics_dir / "sensitivity_covariate_ablation.json")
        oracle_result = _load_json_if_exists(metrics_dir / "sensitivity_oracle.json")
    else:
        horizon_result = (
            _load_json_if_exists(metrics_dir / "sensitivity_horizon.json")
            if args.skip_horizon
            else run_horizon_sensitivity(
                horizons=args.horizons,
                include_finetune=not args.skip_finetune_horizon,
                reuse_finetune_params=args.reuse_finetune_params,
                config=config, logger=logger,
            )
        )
        ablation_result = (
            _load_json_if_exists(metrics_dir / "sensitivity_covariate_ablation.json")
            if args.skip_covariate
            else run_covariate_ablation(config=config, logger=logger)
        )
        oracle_result = (
            _load_json_if_exists(metrics_dir / "sensitivity_oracle.json")
            if args.skip_oracle
            else run_oracle_ablation(config=config, logger=logger)
        )

    summary_md = build_summary_markdown(horizon_result, ablation_result, oracle_result, config)
    summary_path = metrics_dir / "sensitivity_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_md, encoding="utf-8")
    logger.info("Ringkasan sensitivitas tersimpan di %s", summary_path)

    figure_path = plot_sensitivity_summary(
        horizon_result, ablation_result, oracle_result,
        resolve_path(config["paths"]["figures_dir"]) / "sensitivity_summary.png",
        config["visualization"]["figure_dpi"], config,
    )
    logger.info("Figure ringkasan tersimpan di %s", figure_path)

    return {
        "horizon": horizon_result,
        "covariate_ablation": ablation_result,
        "oracle": oracle_result,
        "summary_path": str(summary_path),
        "figure_path": str(figure_path),
    }


if __name__ == "__main__":
    main()
