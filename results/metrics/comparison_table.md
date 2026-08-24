# Tabel Perbandingan Skema

Protokol: rolling origin (expanding window), H = 10 hari perdagangan, stride = 1, 181 jendela pada periode uji (2025-11-11 s.d. 2026-08-03).

Seluruh skema dievaluasi pada **himpunan jendela rolling origin yang persis sama**. Kovariat Emas dan Dollar Index diperlakukan sebagai *past-only*: nilai masa depannya tidak pernah diberikan ke model.

## Tabel 1 — Perbandingan akurasi dan kalibrasi

| Skema | MAE | RMSE | MAPE (%) | MASE | CRPS | nCRPS | Cov80 | Cov95 | Width80 | CrossRate |
|---|---|---|---|---|---|---|---|---|---|---|
| Naive Persistence | 5.6810 | 8.2954 | 7.511 | 1.0000 | 3.6997 | 0.051063 | 0.7680 | 0.9149 | 19.6722 | 0.000000 |
| Chronos-2 Zero-Shot | 6.0303 | 9.1035 | 7.977 | 1.0615 | 3.8774 | 0.053516 | 0.6801 | 0.9094 | 14.8809 | 0.000000 |
| Chronos-2 Fine-Tuned | 6.0581 | 8.9864 | 8.012 | 1.0664 | 3.9578 | 0.054626 | 0.6630 | 0.8751 | 14.8096 | 0.000000 |

Keterangan kolom: MASE = MAE skema / MAE baseline pada jendela yang sama (< 1 berarti mengungguli baseline). nCRPS = CRPS / rata-rata |y|. Cov80 dan Cov95 = cakupan empiris interval 80% dan 95% (target 0,80 dan 0,95). Width80 = lebar rata-rata interval 80% dalam USD — wajib dibaca bersama Cov80, karena cakupan tinggi yang dicapai lewat interval kelewat lebar bukan kalibrasi yang baik. CrossRate = proporsi pelanggaran urutan kuantil.

## Tabel 2 — Uji Diebold-Mariano (rugi = galat absolut, HAC Newey-West)

| Perbandingan | Selisih rugi rata-rata | Statistik DM | p-value | Lag HAC | Signifikan (α = 0.05) | Lebih baik |
|---|---|---|---|---|---|---|
| Chronos-2 Zero-Shot vs Chronos-2 Fine-Tuned **(uji utama)** | -0.027842 | -0.2018 | 0.8401 | 9 | tidak | tidak terbukti berbeda |
| Chronos-2 Zero-Shot vs Naive Persistence | 0.349285 | 1.2688 | 0.2045 | 9 | tidak | tidak terbukti berbeda |
| Chronos-2 Fine-Tuned vs Naive Persistence | 0.377127 | 1.8109 | 0.0702 | 9 | tidak | tidak terbukti berbeda |

H0 uji DM: kedua ramalan sama akuratnya. Statistik negatif berarti skema pertama memiliki rugi lebih kecil. Karena jendela rolling origin saling tumpang tindih, ragam diestimasi dengan HAC Newey-West berlag H − 1 = 9. **p-value di atas α berarti perbedaan angka pada Tabel 1 tidak terbukti secara statistik dan tidak boleh dinarasikan sebagai keunggulan.**
