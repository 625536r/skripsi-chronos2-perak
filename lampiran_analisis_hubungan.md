# Lampiran — Tabel & Grafik Analisis Hubungan Antar-Variabel (Tujuan 1)

> Materi mentah untuk Bab 4 bagian 4.2. Angka diambil langsung dari `results/metrics/relationship_analysis.json` dan `results/metrics/mutual_information.json`, dihitung pada data latih saja. Belum ditempel ke `bab4.md` — pindahkan manual setelah direview.

## 1. Tabel Uji Stasioneritas (ADF & KPSS)

Sumber: `results/metrics/relationship_analysis.json` (key `stationarity`). Regresi deterministik konstanta saja (tanpa tren), α = 0,05.

| Seri | Varian | ADF statistik | ADF p-value | KPSS statistik | KPSS p-value | Simpulan (α = 0,05) |
|---|---|---:|---:|---:|---:|---|
| Perak | Level | −0,692 | 0,849 | 2,782 | 0,010* | Non-stasioner |
| Perak | Log-return | −29,731 | 0,000 | 0,101 | >0,100* | Stasioner |
| Emas | Level | 1,077 | 0,995 | 3,664 | 0,010* | Non-stasioner |
| Emas | Log-return | −11,688 | 0,000 | 0,289 | >0,100* | Stasioner |
| Dollar Index | Level | −2,157 | 0,222 | 1,492 | 0,010* | Non-stasioner |
| Dollar Index | Log-return | −23,268 | 0,000 | 0,150 | >0,100* | Stasioner |

\* KPSS p-value dari statsmodels dipangkas ke batas tabel referensi ([0,01 ; 0,1]) — tulis sebagai "<0,01" / ">0,1" di laporan, bukan nilai eksak. Nilai kritis 5% acuan: ADF ≈ −2,865, KPSS ≈ 0,463 (hampir seragam di ketiga seri karena n besar).

**Simpulan:** ADF dan KPSS sepakat untuk ketiga seri: non-stasioner pada level, stasioner pada log-return → ordo integrasi d = 1 untuk Perak, Emas, dan Dollar Index. Ini yang mendasari `d_max = 1` pada prosedur Toda-Yamamoto (komponen 4) dan alasan ACF/PACF dihitung pada log-return, bukan level.

## 2. Grafik & Angka Korelasi Silang (Cross-Correlation Function / CCF)

Sumber: `results/metrics/relationship_analysis.json` (key `cross_correlation`), log-return data latih, lag −10 s.d. +10. Batas signifikansi (α = 0,05, dua sisi) ≈ ±0,0675 untuk kedua pasangan (n ≈ 868–878).

**Grafik:**

![CCF Perak–Emas](../../results/figures/relationship/ccf_perak_emas.png)

![CCF Perak–Dollar Index](../../results/figures/relationship/ccf_perak_dxy.png)

**Angka CCF Perak–Emas (log-return):**

| Lag | −10 | −9 | −8 | −7 | −6 | −5 | −4 | −3 | −2 | −1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CCF | −0,061 | −0,001 | −0,011 | −0,007 | −0,041 | 0,031 | 0,101* | 0,027 | −0,008 | −0,028 | **0,778*** | −0,018 | −0,037 | 0,019 | 0,025 | 0,039 | −0,030 | 0,008 | −0,034 | 0,009 | −0,057 |

**Angka CCF Perak–Dollar Index (log-return):**

| Lag | −10 | −9 | −8 | −7 | −6 | −5 | −4 | −3 | −2 | −1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CCF | 0,046 | 0,017 | −0,019 | −0,004 | 0,040 | 0,023 | −0,081* | 0,015 | 0,057 | −0,026 | **−0,406*** | −0,029 | −0,011 | 0,016 | −0,026 | −0,039 | 0,041 | −0,019 | −0,032 | −0,040 | 0,053 |

\* signifikan pada α = 0,05.

**Simpulan:** korelasi memuncak tajam di **lag 0** untuk kedua pasangan (Perak–Emas 0,778; Perak–DXY −0,406), jauh lebih besar dari lag manapun selain lag ±4 yang signifikan tapi kecil (≈0,08–0,10, kemungkinan false positive dari banyaknya lag diuji). Pola ini menunjukkan hubungan bersifat **sezaman**, bukan lead-lag yang jelas.

## 3. Tabel Uji Kointegrasi (Engle-Granger & Johansen)

Sumber: `results/metrics/relationship_analysis.json` (key `cointegration`), level, data latih.

**Engle-Granger (per arah regresi):**

| Pasangan (y0 ~ y1) | Statistik EG | p-value | Kointegrasi (α = 0,05) |
|---|---:|---:|---|
| Perak ~ Emas | −3,2826 | 0,0572 | Tidak |
| Emas ~ Perak | −2,6766 | 0,2081 | Tidak |
| Perak ~ DXY | −0,6737 | 0,9494 | Tidak |
| DXY ~ Perak | −2,1267 | 0,4624 | Tidak |
| Emas ~ DXY | 0,3388 | 0,9913 | Tidak |
| DXY ~ Emas | −2,2961 | 0,3755 | Tidak |

**Johansen** (sistem Perak, Emas, DXY; k_ar_diff = 0):

| Uji | Rank kointegrasi terindikasi |
|---|---:|
| Trace test | 1 |
| Max-eigenvalue test | 0 |

**Simpulan:** Engle-Granger tidak menemukan kointegrasi pada pasangan manapun (Perak–Emas paling mendekati, p = 0,057). Johansen ambigu (trace vs max-eigenvalue tidak sepakat). Karena hasil ambigu/negatif ini, uji presedensi prediktif (komponen 4) memakai **Toda-Yamamoto** (robust terhadap ada/tidaknya kointegrasi), bukan VECM.

## 4. Tabel Uji Presedensi Prediktif Toda-Yamamoto

Sumber: `results/metrics/relationship_analysis.json` (key `granger_toda_yamamoto`). Lag VAR akhir p = 3 (naik dari p = 1 hasil AIC/BIC karena eskalasi Ljung-Box pada diagnostik residual), total lag VAR p + d_max = 4, uji Wald dengan kovarians HAC.

| Arah (X → Y) | Lag diuji | Statistik Wald | df | p-value | Signifikan (α = 0,05) |
|---|---|---:|---:|---:|---|
| Emas → Perak | 1–3 | 3,924 | 3 | 0,270 | Tidak |
| DXY → Perak | 1–3 | 1,962 | 3 | 0,580 | Tidak |
| Perak → Emas | 1–3 | 4,440 | 3 | 0,218 | Tidak |
| Perak → DXY | 1–3 | 0,525 | 3 | 0,913 | Tidak |

**Simpulan:** tidak ada satu pun arah yang signifikan pada α = 0,05 — tidak ditemukan presedensi prediktif berlag antar ketiga variabel; konsisten dengan CCF (komponen 2) yang memuncak di lag 0 dan Mutual Information berlag (komponen 5) yang runtuh ke ≈ 0 pada lag ≥ 1.

## 5. Tabel Perbandingan Mutual Information (MI) Berlag: Level vs. Log-Return

Sumber: `results/metrics/mutual_information.json` (key `lagged`), data latih saja, satuan nat, rata-rata 20 pengulangan estimator k-NN.

| Kovariat | Varian | Lag 0 | Lag 1 | Lag 2 | Lag 3 | Lag 5 |
|---|---|---:|---:|---:|---:|---:|
| Emas | Level harga | 1,348 | 1,165 | 1,126 | 1,107 | 1,134 |
| Emas | Log-return | 0,415 | 0,000 | 0,000 | 0,000 | 0,005 |
| Dollar Index | Level harga | 0,775 | 0,693 | 0,634 | 0,661 | 0,676 |
| Dollar Index | Log-return | 0,090 | 0,000 | 0,000 | 0,000 | 0,000 |

**Simpulan:** MI level tetap tinggi di semua lag (didominasi kesamaan tren antar seri non-stasioner, cenderung overestimasi — lihat komponen 1). MI log-return, yang secara statistik sah, hanya tinggi di **lag 0** (Emas 0,415; DXY 0,090) lalu **runtuh ke ≈ 0 pada lag ≥ 1** untuk kedua kovariat. Pola ini sejalan dengan hasil Toda-Yamamoto (komponen 4) yang juga tidak menemukan presedensi berlag — dua metode independen (satu linear, satu non-linear) menunjuk kesimpulan yang sama: keterkaitan Perak dengan Emas/DXY bersifat **sezaman**, bukan berlag.
