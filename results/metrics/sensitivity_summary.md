# Analisis Sensitivitas

Disimpan **terpisah** dari hasil utama (`results/metrics/final_results.json`, H = 10) agar tidak tercampur. Sumber angka: `sensitivity_horizon.json`, `sensitivity_covariate_ablation.json`, `sensitivity_oracle.json`, dan gambar `results/figures/sensitivity_summary.png`.

## 1. Sensitivitas horizon

Protokol rolling origin diulang pada H = [5, 20], dibandingkan dengan H utama = 10. Tiap horizon memakai daftar origin sendiri (jumlah jendela berbeda), sehingga yang dibandingkan adalah metrik teragregasi, bukan jendela per jendela.

> Skema Fine-Tuned pada H selain H utama memakai **ulang** hyperparameter terbaik hasil tuning H utama, **TANPA tuning grid ulang per horizon** (penyimpangan eksplisit dari protokol pemilihan hyperparameter, demi kelayakan komputasi: tuning ulang berarti mengulang seluruh pencarian 18 kandidat untuk tiap horizon -- lihat catatan pada `sensitivity_horizon.json`).

| H | Skema | n jendela | MAE | RMSE | MASE | CRPS | nCRPS | Cov80 | Cov95 |
|---|---|---|---|---|---|---|---|---|---|
| 5 | Naive Persistence | 186 | 4.1635 | 6.5087 | 1.0000 | 2.7490 | 0.038153 | 0.7957 | 0.9161 |
| 5 | Chronos-2 Zero-Shot | 186 | 4.4850 | 7.1476 | 1.0772 | 2.8586 | 0.039674 | 0.7215 | 0.9484 |
| 5 | Chronos-2 Fine-Tuned | 186 | 4.4762 | 7.1046 | 1.0751 | 2.8798 | 0.039968 | 0.7054 | 0.9409 |
| 10 **(H utama)** | Naive Persistence | 181 | 5.6810 | 8.2954 | 1.0000 | 3.6997 | 0.051063 | 0.7680 | 0.9149 |
| 10 **(H utama)** | Chronos-2 Zero-Shot | 181 | 6.0303 | 9.1035 | 1.0615 | 3.8774 | 0.053516 | 0.6801 | 0.9094 |
| 10 **(H utama)** | Chronos-2 Fine-Tuned | 181 | 6.0581 | 8.9864 | 1.0664 | 3.9578 | 0.054626 | 0.6630 | 0.8751 |
| 20 | Naive Persistence | 171 | 7.8951 | 10.8918 | 1.0000 | 5.2532 | 0.071580 | 0.7319 | 0.8839 |
| 20 | Chronos-2 Zero-Shot | 171 | 8.2980 | 11.7910 | 1.0510 | 5.5018 | 0.074969 | 0.6582 | 0.8547 |
| 20 | Chronos-2 Fine-Tuned | 171 | 8.2820 | 11.5829 | 1.0490 | 5.5588 | 0.075746 | 0.6357 | 0.8281 |

**Uji Diebold-Mariano per horizon** (rugi = galat absolut, HAC Newey-West):

| H | Perbandingan | p-value | Signifikan (α=0.05) |
|---|---|---|---|
| 5 | Chronos-2 Zero-Shot vs Chronos-2 Fine-Tuned | 0.8345 | tidak |
| 10 | Chronos-2 Zero-Shot vs Chronos-2 Fine-Tuned | 0.8401 | tidak |
| 20 | Chronos-2 Zero-Shot vs Chronos-2 Fine-Tuned | 0.9406 | tidak |

## 2. Ablasi kovariat (skema terbaik)

Skema dasar: **Chronos-2 Zero-Shot** (terpilih otomatis: nCRPS terkecil pada hasil utama). H = 10, 181 jendela (2025-11-11 s.d. 2026-08-03) -- daftar origin diverifikasi identik dengan hasil utama.

| Kondisi | Kovariat | MAE | CRPS | MASE | nCRPS |
|---|---|---|---|---|---|
| Hanya Perak | (tidak ada) | 6.1992 | 3.9461 | 1.0912 | 0.054464 |
| Perak + Emas | gold_close | 6.2241 | 3.9980 | 1.0956 | 0.055181 |
| Perak + Emas + Dollar Index | gold_close, dxy_close | 6.0303 | 3.8774 | 1.0615 | 0.053516 |

**Uji Diebold-Mariano antar kondisi** (menguji apakah kovariat benar-benar membantu):

| Perbandingan | Selisih rugi rata-rata | p-value | Signifikan (α=0.05) | Lebih baik |
|---|---|---|---|---|
| Hanya Perak vs Perak + Emas + Dollar Index **(uji utama)** | 0.168925 | 0.3320 | tidak | tidak terbukti berbeda |
| Perak + Emas vs Perak + Emas + Dollar Index | 0.193790 | 0.3058 | tidak | tidak terbukti berbeda |
| Hanya Perak vs Perak + Emas | -0.024865 | 0.5823 | tidak | tidak terbukti berbeda |

**Kesimpulan uji utama:** Menambahkan kovariat (Emas + DXY) **tidak terbukti signifikan** mengubah akurasi dibanding hanya Perak (p = 0.3320 >= 0.05); secara empiris kovariat ini tidak terbukti membantu pada protokol evaluasi ini.

**Sambungan dengan Mutual Information:** (sumber: `results/metrics/mutual_information.json`)

- MI (level): gold_close=1.3477, dxy_close=0.7753
- MI (log_return): gold_close=0.4150, dxy_close=0.0899

Bila MI (terutama pada log-return, varian yang secara statistik sah -- lihat Notebook 02) kecil untuk kedua kovariat, hasil ini KONSISTEN dengan uji Diebold-Mariano di atas: kovariat secara empiris tidak banyak menambah informasi bagi ramalan Perak.

## 3. Skenario ORACLE / EX-POST — TIDAK REALISTIS (opsional, batas atas teoretis)

> **EX-POST / TIDAK REALISTIS -- batas atas teoretis. Kovariat Emas/DXY diberi nilai AKTUALnya pada horizon ramalan (known-future), sesuatu yang TIDAK tersedia saat peramalan sungguhan dilakukan. JANGAN dinarasikan sebagai keunggulan model yang dapat dipakai.**

H = 10, 181 jendela (2025-11-11 s.d. 2026-08-03), kovariat known-future: gold_close, dxy_close.

| Skema | MAE | CRPS | Cov80 | Cov95 |
|---|---|---|---|---|
| Chronos-2 Zero-Shot (realistis) | 6.0303 | 3.8774 | 0.6801 | 0.9094 |
| **ORACLE/EX-POST (TIDAK REALISTIS)** | 5.0152 | 3.2792 | 0.7061 | 0.8950 |

Selisih MAE (realistis − oracle): 1.0151 USD (16.83% dari MAE realistis) -- **ini bukan potensi perbaikan yang dapat dicapai model manapun secara realistis**, melainkan ukuran seberapa besar informasi yang (secara ilegitimate) diberikan kepada model pada skenario ex-post ini.

Uji Diebold-Mariano (realistis vs oracle/ex-post): p = 0.0006 (signifikan pada α = 0.05).

---

Gambar ringkasan ketiga analisis: `results/figures/sensitivity_summary.png`.
