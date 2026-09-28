# BAB 3 METODOLOGI PENELITIAN

> Status: KERANGKA — belum dikembangkan jadi paragraf penuh.

## 3.1 Jenis dan Pendekatan Penelitian
- Penelitian kuantitatif, desain eksperimen komparatif pada satu deret waktu.

## 3.2 Data Penelitian
- Sumber: Yahoo Finance via yfinance.
- Objek: Perak (target), Emas dan Dollar Index (kovariat).
- Frekuensi harian, rentang lima tahun, tanggal akhir tetap (reproducible).
- Harga yang dipakai: Close tanpa penyesuaian.

## 3.3 Tahap Pra-pemrosesan
- Penyelarasan tanggal antar-bursa (join, forward fill kovariat, penanganan target).
- Pembagian data kronologis: latih/validasi/uji.

## 3.4 Metodologi Tujuan 1 — Analisis Hubungan Antar-Variabel
- Urutan uji dan alasan urutannya (tiap langkah menentukan validitas langkah berikutnya): stasioneritas → ACF/PACF → cross-correlation → kointegrasi → pemilihan lag VAR → diagnostik residual → kausalitas Granger (Toda-Yamamoto) → Mutual Information berlag.
- Seluruhnya dihitung hanya pada data latih.

## 3.5 Metodologi Tujuan 2 — Peramalan Probabilistik
- Skema yang dibandingkan: zero-shot, fine-tuned, baseline naive persistence.
- Perlakuan kovariat sebagai past-only (bukan known-future) dan alasannya.
- Protokol rolling origin: horizon utama, stride, jumlah jendela.
- Level kuantil yang digunakan.
- Prosedur pemilihan hyperparameter fine-tuning pada data validasi (grid, metrik seleksi).
- Analisis sensitivitas: horizon lain, ablasi kovariat, skenario oracle/ex-post (dan penegasan skenario ini tidak realistis).

## 3.6 Metrik Evaluasi
- Akurasi titik (MAE, RMSE, MAPE, MASE).
- Kalibrasi probabilistik (CRPS, coverage 80%/95%, interval width, quantile crossing rate).
- Uji signifikansi Diebold-Mariano dengan koreksi HAC.

## 3.7 Perangkat Penelitian
- Perangkat keras/lunak, versi library, lingkungan eksekusi.

## 3.8 Alur Penelitian
- Diagram tahapan dari pengumpulan data sampai pelaporan hasil.
