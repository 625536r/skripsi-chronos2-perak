# BAB 2 TINJAUAN PUSTAKA

> Status: KERANGKA — belum dikembangkan jadi paragraf penuh.

## 2.1 Penelitian Terdahulu
- Studi keterkaitan Perak–Emas–Dollar Index (co-movement, lead-lag) sebelumnya.
- Studi peramalan harga logam mulia (model klasik: ARIMA/GARCH/VAR, model ML/DL).
- Studi penerapan time series foundation model (Chronos, TimeGPT, Moirai, TimesFM, dll.) termasuk skenario zero-shot vs fine-tuned.
- Tabel perbandingan posisi penelitian ini terhadap penelitian terdahulu (metode, data, gap yang diisi).

## 2.2 Landasan Teori — Analisis Hubungan Antar-Variabel
- Stasioneritas deret waktu, uji ADF dan KPSS.
- Autokorelasi: ACF/PACF.
- Cross-correlation function.
- Kointegrasi: Engle-Granger, Johansen.
- Kausalitas Granger dan prosedur Toda-Yamamoto (kenapa dipakai, bukan uji Granger standar).
- Diagnostik residual VAR: Ljung-Box, ARCH-LM.
- Mutual Information sebagai ukuran ketergantungan non-linear.

## 2.3 Landasan Teori — Peramalan Probabilistik
- Peramalan titik vs peramalan probabilistik, representasi kuantil.
- Time series foundation model: konsep pretraining lintas-domain, zero-shot inference.
- Chronos-2: arsitektur, dukungan kovariat, cara kerja fine-tuning (LoRA vs full fine-tuning).
- Baseline naive persistence.
- Protokol evaluasi rolling origin (expanding window).
- Metrik akurasi titik: MAE, RMSE, MAPE, MASE.
- Metrik kalibrasi probabilistik: CRPS, coverage, interval width, quantile crossing.
- Uji signifikansi Diebold-Mariano dan koreksi HAC.

## 2.4 Kerangka Berpikir
- Diagram/narasi alur penelitian dari data mentah → dua jalur analisis (hubungan & peramalan) → temuan.
