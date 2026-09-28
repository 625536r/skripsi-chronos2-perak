### Tabel 1. Ordo integrasi tiap seri (data latih)

| Seri | Verdict level | Verdict log-return | Ordo integrasi (d) | Catatan |
|---|---|---|---:|---|
| SILVER | non_stasioner | stasioner | 1 | Level non-stasioner dan log-return stasioner (ADF dan KPSS sepakat pada log-return); d = 1. |
| GOLD | non_stasioner | stasioner | 1 | Level non-stasioner dan log-return stasioner (ADF dan KPSS sepakat pada log-return); d = 1. |
| DXY | non_stasioner | stasioner | 1 | Level non-stasioner dan log-return stasioner (ADF dan KPSS sepakat pada log-return); d = 1. |

### Tabel 2. Kointegrasi Engle-Granger dan Johansen (level, data latih)

| Pasangan (y0~y1) | Statistik EG | p-value EG | Kointegrasi (alfa=0.05) |
|---|---:|---:|---|
| silver_close~gold_close | -3.2826 | 0.0572 | tidak |
| gold_close~silver_close | -2.6766 | 0.2081 | tidak |
| silver_close~dxy_close | -0.6737 | 0.9494 | tidak |
| dxy_close~silver_close | -2.1267 | 0.4624 | tidak |
| gold_close~dxy_close | 0.3388 | 0.9913 | tidak |
| dxy_close~gold_close | -2.2961 | 0.3755 | tidak |

Johansen (sistem SILVER, GOLD, DXY, k_ar_diff=0): rank kointegrasi trace test = 1, rank max-eigenvalue test = 0.

### Tabel 3. Pemilihan lag VAR (level, data latih)

Lag utama (BIC) = 1; lag pemeriksaan ketahanan (AIC) = 1. Lag dinaikkan menjadi 3 pada tahap diagnostik residual karena Ljung-Box menolak H0 pada lag BIC; lag akhir inilah yang dipakai pada uji presedensi prediktif (Tabel 4).

| Lag | AIC | BIC | HQIC | FPE |
|---:|---:|---:|---:|---:|
| 0 | 14.4349 | 14.4515 | 14.4413 | 1857846.2757 |
| 1 | 1.8064 | 1.8728 | 1.8318 | 6.0884 |
| 2 | 1.8205 | 1.9368 | 1.8650 | 6.1751 |
| 3 | 1.8145 | 1.9806 | 1.8781 | 6.1380 |
| 4 | 1.8268 | 2.0428 | 1.9095 | 6.2142 |
| 5 | 1.8266 | 2.0924 | 1.9284 | 6.2131 |
| 6 | 1.8395 | 2.1551 | 1.9603 | 6.2936 |
| 7 | 1.8502 | 2.2156 | 1.9901 | 6.3614 |
| 8 | 1.8607 | 2.2759 | 2.0197 | 6.4286 |
| 9 | 1.8631 | 2.3282 | 2.0412 | 6.4444 |
| 10 | 1.8809 | 2.3958 | 2.0780 | 6.5598 |
| 11 | 1.8912 | 2.4559 | 2.1074 | 6.6281 |
| 12 | 1.9004 | 2.5149 | 2.1357 | 6.6896 |
| 13 | 1.9124 | 2.5768 | 2.1668 | 6.7708 |
| 14 | 1.9193 | 2.6334 | 2.1927 | 6.8176 |
| 15 | 1.9276 | 2.6916 | 2.2201 | 6.8748 |
| 16 | 1.9407 | 2.7545 | 2.2523 | 6.9660 |
| 17 | 1.9376 | 2.8013 | 2.2683 | 6.9451 |
| 18 | 1.9521 | 2.8656 | 2.3018 | 7.0470 |
| 19 | 1.9649 | 2.9283 | 2.3338 | 7.1389 |
| 20 | 1.9726 | 2.9857 | 2.3605 | 7.1944 |

### Tabel 4. Presedensi prediktif Toda-Yamamoto vs Mutual Information berlag

Diuji pada lag akhir p = 3 (lihat catatan Tabel 3), dengan total lag VAR p + d_max = 4 pada langkah Toda-Yamamoto.

| Pasangan (X -> Y) | Lag diuji | Statistik Wald | df | p-value | Kesimpulan (alfa=0.05) | MI (log-return, lag>=1) |
|---|---:|---:|---:|---:|---|---:|
| GOLD -> SILVER | 1-3 | 3.9244 | 3 | 0.2697 | tidak signifikan | 0.0048 |
| DXY -> SILVER | 1-3 | 1.9615 | 3 | 0.5804 | tidak signifikan | 0.0000 |
| SILVER -> GOLD | 1-3 | 4.4399 | 3 | 0.2177 | tidak signifikan | - |
| SILVER -> DXY | 1-3 | 0.5253 | 3 | 0.9133 | tidak signifikan | - |

*Kolom MI diambil dari MI maksimum pada lag >= 1 varian log-return (`results/metrics/mutual_information.json`), sebagai pembanding non-linear terhadap hasil Granger yang bersifat linear. Ketidaksesuaian antara keduanya (mis. MI tinggi tapi Granger tidak signifikan) adalah temuan yang menunjukkan keterkaitan bersifat non-linear atau sezaman, bukan berlag secara linear.*

### Tabel 5. Diagnostik residual VAR (lag akhir p = 3)

| Variabel | Ljung-Box p (lag=10) | Lolos | ARCH-LM p (lag=10) | Lolos |
|---|---:|---|---:|---|
| SILVER | 0.0725 | ya | 0.0000 | tidak |
| GOLD | 0.3774 | ya | 0.0000 | tidak |
| DXY | 0.2926 | ya | 0.0000 | tidak |

Diagnostik keseluruhan: TIDAK lolos.
Lag dinaikkan dari 1 menjadi 3 selama eskalasi Ljung-Box, dan berhasil lolos pada lag 3.
ARCH-LM tetap menolak H0 (heteroskedastisitas bersyarat) pada seluruh variabel meski Ljung-Box sudah lolos -- wajar untuk data finansial dan bukan indikasi kesalahan spesifikasi lag; menambah lag VAR tidak menghilangkan efek ARCH. Uji Wald pada langkah presedensi prediktif sudah memakai kovarians HAC untuk mengantisipasi hal ini.