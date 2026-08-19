"""Uji modul evaluasi dengan kasus sintetis yang jawabannya sudah diketahui.

Strategi pengujian: setiap kasus dibuat sedemikian rupa sehingga nilai metrik
yang benar dapat dihitung tangan lebih dulu, sehingga kegagalan uji menunjuk
langsung ke rumus yang salah — bukan sekadar "angkanya berubah".

Kelompok uji:

* ``TestPointMetrics``        — MAE, RMSE, MAPE, MASE
* ``TestQuantileLoss``        — pinball loss
* ``TestCRPS``                — aproksimasi CRPS dan versi ternormalisasi
* ``TestCoverage``            — cakupan empiris dan lebar interval
* ``TestQuantileCrossing``    — crossing rate untuk kuantil terurut vs teracak
* ``TestDieboldMariano``      — uji signifikansi dengan koreksi HAC
* ``TestAggregateMetrics``    — agregasi per horizon dan keseluruhan
* ``TestBaselineIntegration`` — metrik dihitung terhadap baseline sungguhan
* ``TestValidation``          — penjagaan bentuk array dan masukan tidak valid

Cara menjalankan:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Akar project ditambahkan agar `import src.*` bekerja tanpa instalasi paket
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import (  # noqa: E402
    BaselineError,
    naive_persistence,
    naive_persistence_quantiles,
)
from src.evaluation import (  # noqa: E402
    EvaluationError,
    aggregate_metrics,
    coverage,
    crps_approx,
    diebold_mariano,
    find_quantile_index,
    interval_indices,
    mae,
    mape,
    mase,
    mean_interval_width,
    normalized_crps,
    quantile_crossing_rate,
    quantile_loss,
    rmse,
)

# Level kuantil sesuai spesifikasi penelitian
QUANTILE_LEVELS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975]

# Konfigurasi minimal, agar uji tidak bergantung pada isi config/config.yaml
TEST_CONFIG = {
    "evaluation": {"point_quantile": 0.5, "coverage_levels": [0.80, 0.95]},
    "baseline": {"volatility_window": 60, "min_volatility_window": 20},
}


@pytest.fixture
def rng() -> np.random.Generator:
    """Generator acak dengan seed tetap agar hasil uji reproducible."""
    return np.random.default_rng(42)


def build_quantiles_from_point(
    point: np.ndarray, offsets: list[float] | None = None
) -> np.ndarray:
    """Membangun ramalan kuantil terurut di sekitar sebuah ramalan titik.

    Args:
        point: Ramalan titik ``(n_windows, H)``.
        offsets: Geseran tiap level kuantil. Bila ``None``, dipakai geseran
            simetris yang membuat kuantil 0.5 tepat sama dengan ``point``.

    Returns:
        Array ``(n_windows, H, n_quantiles)`` yang terurut menaik.
    """
    if offsets is None:
        # Simetris terhadap indeks tengah (0.5), menaik tegas
        center = (len(QUANTILE_LEVELS) - 1) / 2
        offsets = [(index - center) for index in range(len(QUANTILE_LEVELS))]

    return point[:, :, None] + np.asarray(offsets, dtype=float)[None, None, :]


# =============================================================================
class TestPointMetrics:
    """Metrik akurasi titik."""

    def test_perfect_prediction_gives_zero(self, rng: np.random.Generator) -> None:
        """Prediksi sempurna wajib menghasilkan seluruh metrik galat = 0."""
        y_true = rng.uniform(20.0, 40.0, size=(12, 10))
        y_pred = y_true.copy()

        assert mae(y_true, y_pred) == 0.0
        assert rmse(y_true, y_pred) == 0.0
        assert mape(y_true, y_pred) == 0.0

    def test_mae_known_value(self) -> None:
        """MAE = rata-rata |galat|, dihitung tangan."""
        y_true = np.array([[10.0, 20.0], [30.0, 40.0]])
        y_pred = np.array([[12.0, 17.0], [30.0, 44.0]])
        # |galat| = 2, 3, 0, 4 -> rata-rata = 9/4 = 2.25
        assert mae(y_true, y_pred) == pytest.approx(2.25)

    def test_rmse_known_value(self) -> None:
        """RMSE = akar rata-rata galat kuadrat, dihitung tangan."""
        y_true = np.array([[0.0, 0.0], [0.0, 0.0]])
        y_pred = np.array([[3.0, 4.0], [0.0, 0.0]])
        # galat kuadrat = 9, 16, 0, 0 -> rata-rata 6.25 -> akar 2.5
        assert rmse(y_true, y_pred) == pytest.approx(2.5)

    def test_rmse_penalizes_outlier_more_than_mae(self) -> None:
        """RMSE harus lebih besar dari MAE saat galat tidak merata."""
        y_true = np.array([[0.0, 0.0, 0.0, 0.0]])
        y_pred = np.array([[0.0, 0.0, 0.0, 8.0]])
        assert rmse(y_true, y_pred) > mae(y_true, y_pred)

    def test_mape_known_value(self) -> None:
        """MAPE dilaporkan dalam persen."""
        y_true = np.array([[100.0, 200.0]])
        y_pred = np.array([[110.0, 180.0]])
        # 10% dan 10% -> 10.0
        assert mape(y_true, y_pred) == pytest.approx(10.0)

    def test_mape_rejects_zero_actual(self) -> None:
        """MAPE tidak terdefinisi bila ada aktual bernilai nol."""
        y_true = np.array([[0.0, 5.0]])
        y_pred = np.array([[1.0, 5.0]])
        with pytest.raises(EvaluationError, match="nol"):
            mape(y_true, y_pred)

    def test_mase_equals_one_when_model_matches_naive(self) -> None:
        """Model yang identik dengan baseline wajib menghasilkan MASE tepat 1."""
        y_true = np.array([[10.0, 12.0], [14.0, 16.0]])
        y_naive = np.array([[9.0, 9.0], [13.0, 13.0]])
        assert mase(y_true, y_naive, y_naive) == pytest.approx(1.0)

    def test_mase_below_one_when_model_better(self) -> None:
        """Model yang lebih akurat dari baseline menghasilkan MASE < 1."""
        y_true = np.array([[10.0, 10.0]])
        y_pred = np.array([[10.5, 10.5]])  # MAE = 0.5
        y_naive = np.array([[12.0, 12.0]])  # MAE = 2.0
        assert mase(y_true, y_pred, y_naive) == pytest.approx(0.25)

    def test_mase_zero_for_perfect_model(self) -> None:
        """Prediksi sempurna menghasilkan MASE = 0."""
        y_true = np.array([[10.0, 10.0]])
        y_naive = np.array([[12.0, 12.0]])
        assert mase(y_true, y_true, y_naive) == 0.0

    def test_mase_rejects_perfect_naive(self) -> None:
        """MASE tidak terdefinisi bila MAE baseline nol."""
        y_true = np.array([[10.0, 10.0]])
        with pytest.raises(EvaluationError, match="tidak terdefinisi"):
            mase(y_true, y_true, y_true)


# =============================================================================
class TestQuantileLoss:
    """Pinball loss."""

    def test_zero_for_perfect_prediction(self) -> None:
        """Ramalan tepat sasaran menghasilkan pinball loss nol pada tiap tau."""
        y_true = np.array([[10.0, 20.0]])
        for tau in QUANTILE_LEVELS:
            assert quantile_loss(y_true, y_true, tau) == pytest.approx(0.0)

    def test_known_value_under_prediction(self) -> None:
        """Under-prediction dihukum dengan bobot tau."""
        y_true = np.array([[10.0]])
        y_pred = np.array([[8.0]])
        # y - yhat = 2 > 0 -> QL = tau * 2 = 0.9 * 2 = 1.8
        assert quantile_loss(y_true, y_pred, 0.9) == pytest.approx(1.8)

    def test_known_value_over_prediction(self) -> None:
        """Over-prediction dihukum dengan bobot (1 - tau)."""
        y_true = np.array([[10.0]])
        y_pred = np.array([[12.0]])
        # y - yhat = -2 < 0 -> QL = (tau - 1) * (-2) = (1 - tau) * 2 = 0.2
        assert quantile_loss(y_true, y_pred, 0.9) == pytest.approx(0.2)

    def test_median_loss_is_half_absolute_error(self) -> None:
        """Pada tau = 0.5, pinball loss tepat setengah galat absolut."""
        y_true = np.array([[10.0, 20.0]])
        y_pred = np.array([[13.0, 16.0]])
        assert quantile_loss(y_true, y_pred, 0.5) == pytest.approx(
            0.5 * mae(y_true, y_pred)
        )

    def test_high_quantile_penalizes_under_prediction_more(self) -> None:
        """Kuantil tinggi lebih menghukum ramalan yang terlalu rendah."""
        y_true = np.array([[10.0]])
        too_low = np.array([[8.0]])
        too_high = np.array([[12.0]])
        assert quantile_loss(y_true, too_low, 0.9) > quantile_loss(
            y_true, too_high, 0.9
        )

    def test_rejects_tau_outside_unit_interval(self) -> None:
        """Level tau di luar (0, 1) ditolak."""
        y_true = np.array([[10.0]])
        with pytest.raises(EvaluationError, match="tau"):
            quantile_loss(y_true, y_true, 1.0)


# =============================================================================
class TestCRPS:
    """Aproksimasi CRPS dari kumpulan kuantil."""

    def test_zero_when_all_quantiles_hit_actual(self) -> None:
        """Sebaran yang runtuh tepat di nilai aktual menghasilkan CRPS = 0."""
        y_true = np.array([[10.0, 20.0]])
        y_pred_quantiles = np.repeat(
            y_true[:, :, None], len(QUANTILE_LEVELS), axis=2
        )
        assert crps_approx(y_true, y_pred_quantiles, QUANTILE_LEVELS) == pytest.approx(
            0.0
        )

    def test_matches_manual_formula(self) -> None:
        """CRPS wajib sama dengan (2/K) * sum QL yang dihitung tangan."""
        levels = [0.25, 0.5, 0.75]
        y_true = np.array([[10.0]])
        y_pred_quantiles = np.array([[[8.0, 9.0, 11.0]]])

        # tau=0.25, d=2  -> 0.25*2  = 0.5
        # tau=0.50, d=1  -> 0.50*1  = 0.5
        # tau=0.75, d=-1 -> 0.25*1  = 0.25
        expected = (2.0 / 3.0) * (0.5 + 0.5 + 0.25)
        assert crps_approx(y_true, y_pred_quantiles, levels) == pytest.approx(expected)

    def test_reduces_to_absolute_error_for_degenerate_forecast(self) -> None:
        """Sebaran satu titik membuat CRPS kembali menjadi galat absolut."""
        levels = [0.25, 0.5, 0.75]
        y_true = np.array([[10.0]])
        y_pred_quantiles = np.full((1, 1, 3), 7.0)

        # Seluruh QL = tau * 3; sum = 3 * (0.25+0.5+0.75) = 4.5; (2/3)*4.5 = 3.0
        assert crps_approx(y_true, y_pred_quantiles, levels) == pytest.approx(3.0)
        assert crps_approx(y_true, y_pred_quantiles, levels) == pytest.approx(
            mae(y_true, np.array([[7.0]]))
        )

    def test_sharper_calibrated_forecast_scores_better(self) -> None:
        """Sebaran yang lebih tajam dan tepat sasaran memberi CRPS lebih kecil."""
        y_true = np.array([[10.0]])
        point = np.array([[10.0]])
        sharp = build_quantiles_from_point(point, offsets=[-0.1, 0.0, 0.1])
        wide = build_quantiles_from_point(point, offsets=[-5.0, 0.0, 5.0])
        levels = [0.25, 0.5, 0.75]

        assert crps_approx(y_true, sharp, levels) < crps_approx(y_true, wide, levels)

    def test_normalized_crps_divides_by_mean_absolute_actual(self) -> None:
        """CRPS ternormalisasi = CRPS dibagi rata-rata |y_true|."""
        levels = [0.25, 0.5, 0.75]
        y_true = np.array([[10.0, 30.0]])  # mean |y| = 20
        y_pred_quantiles = np.repeat(y_true[:, :, None], 3, axis=2)
        y_pred_quantiles = y_pred_quantiles + np.array([-1.0, 0.0, 1.0])

        expected = crps_approx(y_true, y_pred_quantiles, levels) / 20.0
        assert normalized_crps(y_true, y_pred_quantiles, levels) == pytest.approx(
            expected
        )


# =============================================================================
class TestCoverage:
    """Cakupan empiris dan lebar interval."""

    def test_exactly_eighty_percent(self) -> None:
        """Data dirancang agar tepat 80 dari 100 observasi masuk interval."""
        n_windows = 100
        y_true = np.zeros((n_windows, 1))
        lower = np.full((n_windows, 1), -1.0)
        upper = np.full((n_windows, 1), 1.0)

        # 20 observasi sengaja dikeluarkan dari interval
        y_true[:20, 0] = 5.0

        assert coverage(y_true, lower, upper) == pytest.approx(0.80)

    def test_exactly_ninety_five_percent(self) -> None:
        """Kasus serupa untuk target cakupan 95%."""
        n_windows = 200
        y_true = np.zeros((n_windows, 1))
        lower = np.full((n_windows, 1), -1.0)
        upper = np.full((n_windows, 1), 1.0)
        y_true[:10, 0] = -9.0  # 10/200 = 5% di luar interval

        assert coverage(y_true, lower, upper) == pytest.approx(0.95)

    def test_full_and_zero_coverage(self) -> None:
        """Batas ekstrem: semua di dalam (1.0) dan semua di luar (0.0)."""
        y_true = np.zeros((5, 2))
        assert coverage(y_true, y_true - 1.0, y_true + 1.0) == 1.0
        assert coverage(y_true, y_true + 1.0, y_true + 2.0) == 0.0

    def test_bounds_are_inclusive(self) -> None:
        """Observasi yang jatuh tepat di batas dihitung tercakup."""
        y_true = np.array([[1.0, -1.0]])
        lower = np.array([[-1.0, -1.0]])
        upper = np.array([[1.0, 1.0]])
        assert coverage(y_true, lower, upper) == 1.0

    def test_rejects_inverted_interval(self) -> None:
        """Interval terbalik (bawah > atas) ditolak."""
        y_true = np.array([[0.0]])
        with pytest.raises(EvaluationError, match="terbalik"):
            coverage(y_true, np.array([[2.0]]), np.array([[1.0]]))

    def test_mean_interval_width_known_value(self) -> None:
        """Lebar rata-rata dihitung tangan."""
        lower = np.array([[0.0, 1.0]])
        upper = np.array([[2.0, 5.0]])
        # lebar = 2 dan 4 -> rata-rata 3
        assert mean_interval_width(lower, upper) == pytest.approx(3.0)

    def test_wider_interval_gives_higher_coverage(self) -> None:
        """Cakupan tidak boleh turun ketika interval diperlebar."""
        rng = np.random.default_rng(7)
        y_true = rng.normal(size=(50, 3))
        narrow = coverage(y_true, np.full_like(y_true, -0.5), np.full_like(y_true, 0.5))
        wide = coverage(y_true, np.full_like(y_true, -3.0), np.full_like(y_true, 3.0))
        assert wide >= narrow


# =============================================================================
class TestQuantileCrossing:
    """Deteksi quantile crossing."""

    def test_zero_for_sorted_quantiles(self) -> None:
        """Kuantil yang terurut menaik tidak boleh menghasilkan crossing."""
        point = np.full((8, 5), 30.0)
        y_pred_quantiles = build_quantiles_from_point(point)
        assert quantile_crossing_rate(y_pred_quantiles, QUANTILE_LEVELS) == 0.0

    def test_zero_for_equal_quantiles(self) -> None:
        """Kuantil yang sama persis bukan pelanggaran (syaratnya q_i <= q_j)."""
        y_pred_quantiles = np.full((4, 3, len(QUANTILE_LEVELS)), 25.0)
        assert quantile_crossing_rate(y_pred_quantiles, QUANTILE_LEVELS) == 0.0

    def test_positive_for_shuffled_quantiles(self, rng: np.random.Generator) -> None:
        """Kuantil yang sengaja diacak wajib menghasilkan crossing rate > 0."""
        point = np.full((10, 4), 30.0)
        y_pred_quantiles = build_quantiles_from_point(point)

        shuffled = y_pred_quantiles.copy()
        permutation = rng.permutation(len(QUANTILE_LEVELS))
        shuffled = shuffled[:, :, permutation]

        assert quantile_crossing_rate(shuffled, QUANTILE_LEVELS) > 0.0

    def test_exact_rate_for_single_swap(self) -> None:
        """Satu pasangan tertukar memberi rate yang dapat dihitung tangan."""
        n_windows, horizon = 5, 2
        n_quantiles = len(QUANTILE_LEVELS)
        point = np.full((n_windows, horizon), 30.0)
        y_pred_quantiles = build_quantiles_from_point(point)

        # Tukar dua kuantil bersebelahan pada SATU timestep saja.
        # Penukaran ini melanggar 3 pasangan berurutan: (0,1), (1,2), (2,3).
        y_pred_quantiles[0, 0, [1, 2]] = y_pred_quantiles[0, 0, [2, 1]]

        denominator = n_windows * horizon * (n_quantiles - 1)
        assert quantile_crossing_rate(
            y_pred_quantiles, QUANTILE_LEVELS
        ) == pytest.approx(1.0 / denominator)

    def test_fully_reversed_quantiles_gives_rate_one(self) -> None:
        """Kuantil terbalik total melanggar seluruh pasangan berurutan."""
        point = np.full((3, 2), 30.0)
        y_pred_quantiles = build_quantiles_from_point(point)[:, :, ::-1]
        assert quantile_crossing_rate(y_pred_quantiles, QUANTILE_LEVELS) == 1.0

    def test_rejects_unsorted_levels(self) -> None:
        """Daftar level yang tidak menaik ditolak."""
        y_pred_quantiles = np.zeros((2, 2, 3))
        with pytest.raises(EvaluationError, match="menaik"):
            quantile_crossing_rate(y_pred_quantiles, [0.5, 0.1, 0.9])


# =============================================================================
class TestDieboldMariano:
    """Uji Diebold-Mariano dengan koreksi HAC Newey-West."""

    def test_identical_losses_give_zero_statistic(self) -> None:
        """Dua ramalan yang sama persis: statistik 0 dan p-value 1."""
        rng = np.random.default_rng(1)
        losses = rng.uniform(0.5, 2.0, size=(40, 10))

        result = diebold_mariano(losses, losses, h=10)

        assert result["mean_loss_differential"] == pytest.approx(0.0)
        assert result["statistic"] == pytest.approx(0.0)
        assert result["p_value"] == pytest.approx(1.0)
        assert result["better"] == "seri"
        assert not result["significant"]

    def test_hac_lag_is_horizon_minus_one(self) -> None:
        """Lag HAC wajib h - 1 sesuai rancangan uji Diebold-Mariano."""
        rng = np.random.default_rng(2)
        a = rng.uniform(size=(30, 5))
        b = rng.uniform(size=(30, 5))
        assert diebold_mariano(a, b, h=5)["hac_lag"] == 4
        assert diebold_mariano(a, b, h=1)["hac_lag"] == 0

    def test_detects_clearly_worse_forecast(self) -> None:
        """Selisih rugi yang konsisten dan besar wajib terdeteksi signifikan."""
        rng = np.random.default_rng(3)
        loss_b = rng.uniform(1.0, 1.2, size=60)
        loss_a = loss_b + 1.0  # A konsisten lebih buruk

        result = diebold_mariano(loss_a, loss_b, h=10)

        assert result["statistic"] > 0  # positif berarti B lebih baik
        assert result["p_value"] < 0.05
        assert result["significant"]
        assert result["better"] == "b"

    def test_sign_convention_when_a_is_better(self) -> None:
        """Statistik negatif menandakan ramalan A lebih akurat."""
        rng = np.random.default_rng(4)
        loss_a = rng.uniform(1.0, 1.2, size=60)
        loss_b = loss_a + 1.0

        result = diebold_mariano(loss_a, loss_b, h=10)

        assert result["statistic"] < 0
        assert result["better"] == "a"

    def test_no_difference_is_not_significant(self) -> None:
        """Dua ramalan yang sama baiknya tidak boleh dinyatakan berbeda."""
        rng = np.random.default_rng(5)
        loss_a = rng.normal(1.0, 0.1, size=200)
        loss_b = rng.normal(1.0, 0.1, size=200)

        assert not diebold_mariano(loss_a, loss_b, h=10)["significant"]

    def test_hac_widens_variance_under_positive_autocorrelation(self) -> None:
        """Koreksi HAC harus memperkecil |statistik| saat selisih rugi berkorelasi.

        Inilah alasan koreksi HAC wajib dipakai pada jendela yang tumpang
        tindih: tanpa koreksi, ragamnya diremehkan dan uji jadi terlalu mudah
        menyatakan signifikan.
        """
        rng = np.random.default_rng(6)
        n = 300
        noise = rng.normal(size=n)
        # Deret berautokorelasi positif kuat (rata-rata bergerak 10 langkah)
        differential = 0.05 + np.convolve(noise, np.ones(10) / 10, mode="same")

        zero = np.zeros(n)
        without_hac = diebold_mariano(differential, zero, h=1)["statistic"]
        with_hac = diebold_mariano(differential, zero, h=10)["statistic"]

        assert abs(with_hac) < abs(without_hac)

    def test_rejects_too_few_windows(self) -> None:
        """Lag HAC tidak boleh melebihi jumlah observasi yang tersedia."""
        with pytest.raises(EvaluationError, match="melebihi"):
            diebold_mariano(np.zeros(5), np.ones(5), h=20)

    def test_rejects_mismatched_shapes(self) -> None:
        """Bentuk kedua masukan wajib sama."""
        with pytest.raises(EvaluationError, match="tidak sama"):
            diebold_mariano(np.zeros((10, 3)), np.zeros((10, 4)), h=3)


# =============================================================================
class TestAggregateMetrics:
    """Agregasi metrik per horizon dan keseluruhan."""

    def test_perfect_prediction_zeroes_every_metric(self) -> None:
        """Prediksi sempurna: seluruh metrik galat nol, cakupan penuh."""
        n_windows, horizon = 6, 10
        y_true = np.full((n_windows, horizon), 30.0)
        y_pred_quantiles = np.repeat(
            y_true[:, :, None], len(QUANTILE_LEVELS), axis=2
        )

        result = aggregate_metrics(
            y_true, y_pred_quantiles, QUANTILE_LEVELS, config=TEST_CONFIG
        )
        overall = result["overall"]

        assert overall["mae"] == 0.0
        assert overall["rmse"] == 0.0
        assert overall["mape"] == 0.0
        assert overall["crps"] == 0.0
        assert overall["normalized_crps"] == 0.0
        assert overall["mae_std_windows"] == 0.0
        assert overall["quantile_crossing_rate"] == 0.0
        assert overall["coverage"]["0.8"] == 1.0
        assert overall["coverage"]["0.95"] == 1.0
        assert overall["interval_width"]["0.8"] == 0.0

    def test_structure_has_one_entry_per_horizon(self) -> None:
        """per_horizon wajib berisi tepat H entri dengan h = 1..H berurutan."""
        n_windows, horizon = 5, 10
        rng = np.random.default_rng(11)
        y_true = rng.uniform(25.0, 35.0, size=(n_windows, horizon))
        y_pred_quantiles = build_quantiles_from_point(y_true)

        result = aggregate_metrics(
            y_true, y_pred_quantiles, QUANTILE_LEVELS, config=TEST_CONFIG
        )

        assert result["n_windows"] == n_windows
        assert result["horizon"] == horizon
        assert len(result["per_horizon"]) == horizon
        assert [entry["h"] for entry in result["per_horizon"]] == list(
            range(1, horizon + 1)
        )

    def test_overall_mae_is_mean_of_per_horizon_mae(self) -> None:
        """Konsistensi internal: MAE keseluruhan = rata-rata MAE tiap horizon."""
        rng = np.random.default_rng(12)
        y_true = rng.uniform(25.0, 35.0, size=(7, 10))
        y_pred_quantiles = build_quantiles_from_point(y_true + 0.4)

        result = aggregate_metrics(
            y_true, y_pred_quantiles, QUANTILE_LEVELS, config=TEST_CONFIG
        )
        per_horizon_mae = [entry["mae"] for entry in result["per_horizon"]]

        assert result["overall"]["mae"] == pytest.approx(np.mean(per_horizon_mae))

    def test_std_across_windows_is_zero_for_identical_windows(self) -> None:
        """Bila seluruh jendela identik, simpangan baku antar jendela = 0."""
        single_window = np.arange(1, 11, dtype=float).reshape(1, 10) + 30.0
        y_true = np.repeat(single_window, 8, axis=0)
        y_pred_quantiles = build_quantiles_from_point(y_true + 0.5)

        result = aggregate_metrics(
            y_true, y_pred_quantiles, QUANTILE_LEVELS, config=TEST_CONFIG
        )

        assert result["overall"]["mae_std_windows"] == pytest.approx(0.0)
        assert result["overall"]["crps_std_windows"] == pytest.approx(0.0)

    def test_std_across_windows_is_positive_when_windows_differ(self) -> None:
        """Jendela dengan performa berbeda menghasilkan simpangan baku > 0."""
        # Nilai aktual sengaja bukan nol: MAPE tidak terdefinisi pada aktual nol
        y_true = np.full((3, 2), 100.0)
        point = np.array([[100.0, 100.0], [101.0, 101.0], [105.0, 105.0]])
        y_pred_quantiles = build_quantiles_from_point(point)

        result = aggregate_metrics(
            y_true, y_pred_quantiles, QUANTILE_LEVELS, config=TEST_CONFIG
        )

        assert result["overall"]["mae_std_windows"] > 0.0

    def test_mase_included_only_when_naive_supplied(self) -> None:
        """MASE muncul hanya bila ramalan baseline diberikan."""
        y_true = np.full((4, 3), 20.0)
        y_pred_quantiles = build_quantiles_from_point(y_true + 0.2)
        y_naive = np.full((4, 3), 21.0)

        without_naive = aggregate_metrics(
            y_true, y_pred_quantiles, QUANTILE_LEVELS, config=TEST_CONFIG
        )
        with_naive = aggregate_metrics(
            y_true,
            y_pred_quantiles,
            QUANTILE_LEVELS,
            y_naive=y_naive,
            config=TEST_CONFIG,
        )

        assert "mase" not in without_naive["overall"]
        assert with_naive["overall"]["mase"] == pytest.approx(0.2)
        assert all("mase" in entry for entry in with_naive["per_horizon"])

    def test_coverage_eighty_percent_end_to_end(self) -> None:
        """Cakupan 80% yang dirancang persis, lewat jalur aggregate_metrics."""
        n_windows = 100
        # Aktual dipusatkan di 100 (bukan nol) agar MAPE tetap terdefinisi
        y_true = np.full((n_windows, 1), 100.0)
        y_true[:20, 0] = 200.0  # 20% sengaja jatuh di luar interval

        point = np.full((n_windows, 1), 100.0)
        y_pred_quantiles = build_quantiles_from_point(point)

        result = aggregate_metrics(
            y_true, y_pred_quantiles, QUANTILE_LEVELS, config=TEST_CONFIG
        )

        assert result["overall"]["coverage"]["0.8"] == pytest.approx(0.80)

    def test_rejects_shape_mismatch(self) -> None:
        """Ketidakcocokan bentuk y_true dan y_pred_quantiles ditolak."""
        y_true = np.zeros((4, 3))
        y_pred_quantiles = np.zeros((4, 5, len(QUANTILE_LEVELS)))
        with pytest.raises(EvaluationError, match="tidak"):
            aggregate_metrics(
                y_true, y_pred_quantiles, QUANTILE_LEVELS, config=TEST_CONFIG
            )


# =============================================================================
class TestBaselineIntegration:
    """Metrik diuji terhadap baseline naive persistence yang sesungguhnya."""

    def test_naive_persistence_repeats_last_value(self) -> None:
        """Ramalan titik baseline adalah nilai terakhir yang diulang H kali."""
        context = np.array([10.0, 11.0, 12.5])
        forecast = naive_persistence(context, horizon=4)

        assert forecast.shape == (4,)
        assert np.all(forecast == 12.5)

    def test_naive_persistence_rejects_empty_context(self) -> None:
        """Konteks kosong ditolak."""
        with pytest.raises(BaselineError, match="kosong"):
            naive_persistence([], horizon=3)

    def test_baseline_median_equals_point_forecast(self) -> None:
        """Median sebaran random walk wajib identik dengan ramalan titik."""
        rng = np.random.default_rng(21)
        context = 30.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=200)))

        point = naive_persistence(context, horizon=10)
        quantiles = naive_persistence_quantiles(
            context, 10, QUANTILE_LEVELS, config=TEST_CONFIG
        )
        median_index = find_quantile_index(QUANTILE_LEVELS, 0.5)

        assert np.allclose(quantiles[:, median_index], point)

    def test_baseline_quantiles_never_cross(self) -> None:
        """Sebaran baseline dibangun monoton, jadi crossing rate wajib nol."""
        rng = np.random.default_rng(22)
        context = 30.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=200)))

        quantiles = naive_persistence_quantiles(
            context, 10, QUANTILE_LEVELS, config=TEST_CONFIG
        )
        assert quantile_crossing_rate(quantiles[None, :, :], QUANTILE_LEVELS) == 0.0

    def test_baseline_interval_widens_with_horizon(self) -> None:
        """Ketidakpastian random walk tumbuh sebanding akar horizon."""
        rng = np.random.default_rng(23)
        context = 30.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=200)))

        quantiles = naive_persistence_quantiles(
            context, 10, QUANTILE_LEVELS, config=TEST_CONFIG
        )
        lower_index, upper_index = interval_indices(QUANTILE_LEVELS, 0.80)
        widths = quantiles[:, upper_index] - quantiles[:, lower_index]

        assert np.all(np.diff(widths) > 0)
        assert widths[-1] > widths[0]

    def test_baseline_mase_against_itself_is_one(self) -> None:
        """Baseline yang dievaluasi terhadap dirinya sendiri memberi MASE = 1."""
        rng = np.random.default_rng(24)
        values = 30.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=300)))
        horizon, n_windows = 10, 20

        y_true = np.empty((n_windows, horizon))
        y_naive = np.empty((n_windows, horizon))
        y_quantiles = np.empty((n_windows, horizon, len(QUANTILE_LEVELS)))

        for index in range(n_windows):
            origin = 200 + index
            context = values[:origin]
            y_true[index] = values[origin : origin + horizon]
            y_naive[index] = naive_persistence(context, horizon)
            y_quantiles[index] = naive_persistence_quantiles(
                context, horizon, QUANTILE_LEVELS, config=TEST_CONFIG
            )

        result = aggregate_metrics(
            y_true, y_quantiles, QUANTILE_LEVELS, y_naive=y_naive, config=TEST_CONFIG
        )

        assert result["overall"]["mase"] == pytest.approx(1.0)
        assert result["overall"]["quantile_crossing_rate"] == 0.0
        assert 0.0 <= result["overall"]["coverage"]["0.95"] <= 1.0

    def test_baseline_rejects_non_positive_prices(self) -> None:
        """Harga <= 0 membuat log-return tidak terdefinisi dan wajib ditolak."""
        with pytest.raises(BaselineError, match="log-return"):
            naive_persistence_quantiles(
                np.array([1.0, 0.0, 2.0] * 20), 5, QUANTILE_LEVELS, config=TEST_CONFIG
            )


# =============================================================================
class TestValidation:
    """Penjagaan bentuk array dan pencarian indeks kuantil."""

    def test_rejects_one_dimensional_input(self) -> None:
        """Masukan 1-D ditolak: konvensi wajib (n_windows, H)."""
        with pytest.raises(EvaluationError, match="n_windows"):
            mae(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_rejects_nan(self) -> None:
        """NaN pada masukan ditolak agar tidak menghasilkan metrik menyesatkan."""
        with pytest.raises(EvaluationError, match="NaN"):
            mae(np.array([[1.0, np.nan]]), np.array([[1.0, 2.0]]))

    def test_find_quantile_index(self) -> None:
        """Pencarian indeks level kuantil."""
        assert find_quantile_index(QUANTILE_LEVELS, 0.5) == 6
        assert find_quantile_index(QUANTILE_LEVELS, 0.025) == 0
        assert find_quantile_index(QUANTILE_LEVELS, 0.975) == 12

    def test_find_quantile_index_missing_level(self) -> None:
        """Level yang tidak tersedia menimbulkan galat yang jelas."""
        with pytest.raises(EvaluationError, match="tidak ada"):
            find_quantile_index(QUANTILE_LEVELS, 0.33)

    def test_interval_indices_match_decision_d5(self) -> None:
        """Interval 80% memakai 0.1/0.9 dan 95% memakai 0.025/0.975."""
        assert interval_indices(QUANTILE_LEVELS, 0.80) == (2, 10)
        assert interval_indices(QUANTILE_LEVELS, 0.95) == (0, 12)
