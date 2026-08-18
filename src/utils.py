"""Utilitas bersama untuk seluruh modul pipeline.

Berisi fungsi dasar yang dipakai berulang oleh modul lain: pemuatan konfigurasi,
penyiapan logger, penguncian seed, penyimpanan JSON, dan pencatatan versi
lingkungan kerja.

Modul ini sengaja tidak memuat logika data maupun model.

Cara menjalankan mandiri (mencatat versi lingkungan ke results/environment.json):
    python -m src.utils
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Akar project = satu tingkat di atas folder src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Lokasi konfigurasi bawaan
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Daftar library yang versinya wajib dicatat (nama modul untuk import)
TRACKED_LIBRARIES: tuple[str, ...] = (
    "autogluon.timeseries",
    "chronos",
    "torch",
    "yfinance",
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "yaml",
    "jupyterlab",
    "ipykernel",
)


def resolve_path(relative_path: str | os.PathLike[str]) -> Path:
    """Mengubah path relatif pada config menjadi path absolut terhadap akar project.

    Path yang sudah absolut dikembalikan apa adanya.

    Args:
        relative_path: Path relatif (mis. ``"data/raw"``) atau absolut.

    Returns:
        Path absolut.
    """
    path = Path(relative_path)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Memuat berkas konfigurasi YAML.

    Args:
        config_path: Lokasi berkas konfigurasi. Bila ``None``, dipakai
            ``config/config.yaml`` pada akar project.

    Returns:
        Isi konfigurasi sebagai dictionary.

    Raises:
        FileNotFoundError: Bila berkas konfigurasi tidak ditemukan.
        ValueError: Bila isi berkas kosong atau bukan berupa mapping.
    """
    path = resolve_path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    if not path.is_file():
        raise FileNotFoundError(f"Berkas konfigurasi tidak ditemukan: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Isi konfigurasi tidak valid (bukan mapping): {path}")

    return config


def setup_logger(
    name: str,
    log_dir: str | os.PathLike[str] = "logs",
    level: int = logging.INFO,
    to_console: bool = True,
) -> logging.Logger:
    """Menyiapkan logger yang menulis ke berkas di ``logs/`` sekaligus ke konsol.

    Berkas log diberi nama ``<name>.log``. Pemanggilan berulang dengan ``name``
    yang sama tidak menggandakan handler.

    Args:
        name: Nama logger, sekaligus nama berkas log.
        log_dir: Direktori penyimpanan berkas log.
        level: Level logging minimum.
        to_console: Bila ``True``, log juga dicetak ke stdout.

    Returns:
        Objek logger yang siap dipakai.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Cegah pesan diteruskan ke root logger (menghindari duplikasi output)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_directory = resolve_path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(
        log_directory / f"{name}.log", mode="a", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if to_console:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def set_all_seeds(seed: int = 42) -> int:
    """Mengunci seed acak pada python, numpy, dan torch demi reproducibility.

    Library yang belum terpasang dilewati tanpa menimbulkan error.

    Args:
        seed: Nilai seed yang dipakai.

    Returns:
        Nilai seed yang diterapkan.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed


def save_json(data: Any, file_path: str | os.PathLike[str], indent: int = 2) -> Path:
    """Menyimpan objek python ke berkas JSON.

    Direktori tujuan dibuat otomatis bila belum ada. Objek yang tidak dapat
    diserialisasi secara langsung (mis. numpy scalar, Path, Timestamp) diubah
    menjadi string.

    Args:
        data: Objek yang akan disimpan.
        file_path: Lokasi berkas tujuan.
        indent: Lebar indentasi JSON.

    Returns:
        Path absolut berkas yang tersimpan.
    """
    path = resolve_path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False, default=str)

    return path


def _get_library_version(module_name: str) -> str:
    """Mengambil versi sebuah library, atau penanda bila belum terpasang.

    Args:
        module_name: Nama modul yang akan diimpor.

    Returns:
        String versi, ``"NOT INSTALLED"``, atau ``"UNKNOWN"`` bila modul ada
        tetapi tidak memiliki atribut versi.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return "NOT INSTALLED"

    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)

    # Sebagian paket (mis. namespace package) tidak punya __version__
    try:
        from importlib.metadata import PackageNotFoundError, version as pkg_version

        try:
            return str(pkg_version(module_name.replace(".", "-")))
        except PackageNotFoundError:
            return "UNKNOWN"
    except ImportError:
        return "UNKNOWN"


def _get_torch_device_info() -> dict[str, Any]:
    """Mengumpulkan informasi perangkat komputasi torch (CPU/GPU).

    Returns:
        Dictionary informasi perangkat. Berisi penanda bila torch belum terpasang.
    """
    try:
        import torch
    except ImportError:
        return {"torch_available": False}

    info: dict[str, Any] = {
        "torch_available": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if info["cuda_available"]:
        info["device_names"] = [
            torch.cuda.get_device_name(i) for i in range(info["device_count"])
        ]

    return info


def log_environment(
    config: dict[str, Any] | None = None,
    output_path: str | os.PathLike[str] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Mencatat versi python dan seluruh library ke ``results/environment.json``.

    Dipanggil di awal pipeline agar setiap hasil eksperimen dapat ditelusuri
    kembali ke lingkungan kerja yang menghasilkannya.

    Args:
        config: Konfigurasi project. Bila ``None``, dimuat dari lokasi bawaan.
        output_path: Lokasi berkas keluaran. Bila ``None``, diambil dari
            ``paths.environment_file`` pada konfigurasi.
        logger: Logger yang dipakai. Bila ``None``, dibuat logger baru.

    Returns:
        Dictionary informasi lingkungan yang tersimpan.
    """
    if config is None:
        config = load_config()

    if logger is None:
        logger = setup_logger(
            "utils", log_dir=config.get("paths", {}).get("logs_dir", "logs")
        )

    if output_path is None:
        output_path = config.get("paths", {}).get(
            "environment_file", "results/environment.json"
        )

    libraries = {name: _get_library_version(name) for name in TRACKED_LIBRARIES}

    environment: dict[str, Any] = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "libraries": libraries,
        "torch_device": _get_torch_device_info(),
        "seed": config.get("seed"),
    }

    saved_path = save_json(environment, output_path)

    logger.info("Python %s (%s)", environment["python"]["version"], sys.executable)
    for library_name, library_version in libraries.items():
        logger.info("  %-22s %s", library_name, library_version)
    logger.info("Informasi lingkungan tersimpan di: %s", saved_path)

    return environment


if __name__ == "__main__":
    log_environment()
