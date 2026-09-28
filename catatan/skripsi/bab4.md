# BAB 4 HASIL DAN PEMBAHASAN

> Status: KERANGKA — belum dikembangkan jadi paragraf penuh. Bab ini yang paling bergantung pada angka final hasil eksperimen — isi setelah hasil dianggap final.

## 4.1 Deskripsi Data
- Ringkasan deret setelah pra-pemrosesan (jumlah baris, jumlah forward fill, ukuran tiap split).

## 4.2 Hasil Tujuan 1 — Analisis Hubungan Antar-Variabel
- Hasil uji stasioneritas (level vs log-return, ordo integrasi).
- Hasil ACF/PACF.
- Hasil cross-correlation function per pasangan variabel.
- Hasil uji kointegrasi.
- Hasil pemilihan lag VAR dan diagnostik residual.
- Hasil kausalitas Granger (Toda-Yamamoto) tiap arah, tiap pasangan.
- Hasil Mutual Information (level vs log-return) dan perbandingannya dengan Granger.
- Pembahasan: sintesis temuan — apakah co-movement sezaman atau ada presedensi; kesesuaian/ketidaksesuaian Granger vs MI dibahas sebagai temuan, bukan kegagalan.

## 4.3 Hasil Tujuan 2 — Peramalan Probabilistik
- Perbandingan skema pada periode uji: akurasi titik dan kalibrasi probabilistik.
- Hasil uji Diebold-Mariano antar skema.
- Pembahasan hasil fine-tuning vs zero-shot, termasuk indikasi overfitting bila ditemukan.
- Pembahasan kualitas kalibrasi interval.
- Pola metrik per horizon (h=1 s.d. h=H).

## 4.4 Analisis Sensitivitas
- Horizon lain.
- Ablasi kovariat.
- Skenario oracle/ex-post sebagai batas atas teoretis (bukan performa realistis).

## 4.5 Pembahasan Umum
- Keterkaitan temuan Tujuan 1 dan Tujuan 2 (tanpa menggabungkan metodologinya) — misalnya apakah sifat keterkaitan antar-variabel dari Tujuan 1 konsisten dengan manfaat/tidaknya kovariat pada Tujuan 2.
- Keterbatasan yang terlihat dari hasil (rezim pasar, ukuran data, dsb.).
