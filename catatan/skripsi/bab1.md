# BAB 1 PENDAHULUAN

**Implementasi Time Series Foundation Model Chronos-2 untuk Peramalan Probabilistik Harga Perak dengan Variabel Kovariat Harga Emas dan Dollar Index**

> Status: KERANGKA — belum dikembangkan jadi paragraf penuh. Isi tiap bagian setelah kerangka ini disepakati final.

## 1.1 Latar Belakang

Alur narasi (umum ke spesifik):

1. **Pentingnya logam mulia sebagai instrumen investasi/lindung nilai.** Perak sebagai komoditas dengan karakter ganda (logam industri sekaligus safe haven), volatilitasnya lebih tinggi dari emas.
2. **Keterkaitan Perak–Emas–Dollar Index sebagai fenomena yang diyakini luas tapi perlu diuji secara statistik.** Rasio emas-perak dan hubungan harga logam mulia dengan indeks dolar AS sering diasumsikan dalam literatur finansial, tapi arah keterkaitan (siapa mendahului siapa, atau apakah keduanya sekadar bergerak bersamaan) perlu dibuktikan dengan uji formal, bukan diasumsikan.
3. **Keterbatasan pendekatan peramalan konvensional untuk deret harga finansial**: model statistik klasik (ARIMA/GARCH/VAR) butuh estimasi ulang per aset dan sering hanya menghasilkan prediksi titik, padahal keputusan finansial butuh gambaran ketidakpastian (distribusi), bukan angka tunggal.
4. **Munculnya time series foundation model** sebagai paradigma baru: model yang dilatih pada korpus deret waktu besar dan lintas domain, bisa dipakai langsung (zero-shot) tanpa retraining per deret. Perkenalkan Chronos-2 secara singkat: arsitektur, kemampuan native menghasilkan keluaran probabilistik (kuantil), dan dukungan variabel kovariat.
5. **Celah penelitian (research gap)** yang diisi:
   - Sejauh apa keterkaitan Perak–Emas–DXY ini nyata secara statistik dan pada lag berapa — jarang dijawab dengan rangkaian uji lengkap (stasioneritas → kointegrasi → presedensi prediktif → keterkaitan non-linear) sekaligus; kebanyakan studi berhenti di korelasi sederhana.
   - Apakah fine-tuning sebuah foundation model besar benar-benar menguntungkan ketika hanya tersedia **satu deret waktu** (bukan ratusan/ribuan deret seperti skenario pelatihan foundation model pada umumnya) — pertanyaan praktis yang jawabannya belum jelas di literatur, karena kebanyakan evaluasi foundation model dilakukan pada benchmark multi-seri.
6. **Posisi kontribusi skripsi**: menjawab kedua celah di atas pada kasus konkret harga Perak dengan kovariat Emas dan Dollar Index, memakai lima tahun data harian.

Catatan penulisan: boleh menyebut fakta umum penelitian (rentang data, horizon, jumlah skema yang dibandingkan) sebagai gambaran umum, tapi **jangan** mengutip angka hasil akhir (MAE, p-value, dsb.) di bagian ini — itu tempatnya di Bab 4/5.

## 1.2 Rumusan Masalah

Terkait analisis hubungan antar-variabel:
1. Apakah harga Perak, Emas, dan Dollar Index bergerak bersama (co-movement) secara statistik?
2. Pada lag berapa keterkaitan antar ketiga variabel tersebut paling kuat?
3. Apakah terdapat presedensi prediktif (lead-lag) antara Emas/Dollar Index terhadap Perak, atau sebaliknya?

Terkait peramalan probabilistik:
4. Bagaimana kinerja Chronos-2 skema zero-shot dalam meramalkan harga Perak secara probabilistik dengan Emas dan Dollar Index sebagai kovariat, dibandingkan baseline naive persistence?
5. Apakah fine-tuning Chronos-2 pada data historis Perak memberikan perbaikan akurasi dan kalibrasi probabilistik yang signifikan dibandingkan skema zero-shot?
6. Bagaimana kualitas kalibrasi interval prediksi (coverage, lebar interval) yang dihasilkan Chronos-2?

## 1.3 Tujuan Penelitian

Berpasangan langsung dengan rumusan masalah (tujuan 1–3 menjawab pertanyaan 1–3, tujuan 4–6 menjawab pertanyaan 4–6). Tegaskan di paragraf pembuka bahwa penelitian memiliki **dua tujuan yang setara kedudukannya, dengan metodologi berbeda** — bukan satu tujuan sebagai pendukung yang lain.

1. Menyelidiki ko-pergerakan statistik antara harga Perak, Emas, dan Dollar Index.
2. Mengidentifikasi struktur lag keterkaitan antar ketiga variabel.
3. Menguji arah presedensi prediktif antar variabel menggunakan uji kausalitas Granger dan melengkapinya dengan ukuran ketergantungan non-linear.
4. Membangun dan mengevaluasi model peramalan probabilistik harga Perak berbasis Chronos-2 skema zero-shot, dengan Emas dan Dollar Index sebagai kovariat.
5. Membandingkan kinerja skema zero-shot dan fine-tuned Chronos-2 secara statistik.
6. Mengevaluasi kualitas kalibrasi probabilistik (coverage interval, lebar interval) dari peramalan yang dihasilkan.

## 1.4 Manfaat Penelitian

**Manfaat teoritis:**
- Menambah bukti empiris tentang struktur keterkaitan antar-pasar logam mulia dan indeks dolar AS di periode data terkini.
- Menyediakan studi kasus terukur tentang kapan fine-tuning sebuah time series foundation model besar bermanfaat dan kapan tidak, pada skenario deret tunggal — relevan bagi siapa pun yang ingin menerapkan foundation model pada domain dengan data terbatas, tidak hanya komoditas.

**Manfaat praktis:**
- Memberi gambaran ketidakpastian (bukan sekadar titik ramalan) yang lebih relevan untuk pengambilan keputusan finansial berbasis risiko.
- Menjadi referensi metodologis bagi penelitian lain yang ingin menerapkan foundation model time series dengan kovariat pada deret finansial.

Hindari klaim manfaat sebagai "alat rekomendasi investasi" — penelitian ini eksplisit tidak menghasilkan sinyal beli/tahan/jual.

## 1.5 Batasan Masalah

**Cakupan data:**
- Objek: harga Perak (Silver Futures) sebagai target; Emas (Gold Futures) dan Dollar Index sebagai kovariat.
- Sumber data: Yahoo Finance, harga penutupan harian, rentang lima tahun dengan tanggal akhir tetap (bukan data real-time/live).
- Satu aset, satu deret waktu — bukan portofolio multi-aset.

**Cakupan metode analisis hubungan:**
- Uji kausalitas Granger yang digunakan mengidentifikasi presedensi prediktif statistik, **bukan** hubungan sebab-akibat kausal dalam pengertian kausalitas struktural/eksperimental.

**Cakupan metode peramalan:**
- Model yang digunakan hanya Chronos-2 (tidak membandingkan dengan foundation model time series lain seperti TimeGPT/Moirai/TimesFM), dibandingkan terhadap satu baseline (naive persistence).
- Horizon peramalan utama 10 hari perdagangan, dengan pengujian sensitivitas pada horizon lain sebagai pelengkap, bukan temuan utama.
- Kovariat diperlakukan sebagai informasi masa lalu saja (nilai kovariat pada periode yang diramalkan tidak digunakan), karena pada praktiknya nilai Emas dan Dollar Index di masa depan memang tidak diketahui saat peramalan dilakukan.
- Protokol evaluasi menggunakan jendela bergulir (rolling origin) pada satu periode uji historis tertentu; hasil kuantitatif spesifik terhadap rezim pasar pada periode tersebut dan tidak diklaim berlaku umum untuk periode lain.

**Cakupan keluaran:**
- Keluaran penelitian berupa distribusi prediksi (interval/kuantil harga), **bukan** sinyal atau rekomendasi keputusan investasi.

## 1.6 Sistematika Penulisan

Paragraf singkat standar, ringkas isi Bab 1–5 (Pendahuluan, Tinjauan Pustaka, Metodologi, Hasil dan Pembahasan, Penutup) dalam satu paragraf per bab.

---

**Catatan pemakaian kerangka:** bagian yang paling perlu dikembangkan dengan kutipan literatur adalah 1.1 poin 2–4 (co-movement logam mulia, kelemahan model klasik, foundation model time series) — biasanya diminta dosen pembimbing untuk didukung sitasi jurnal, bukan hanya deskripsi sistem.

**Catatan tambahan:** versi 1.1 yang sudah dikembangkan jadi paragraf penuh dengan
sitasi ada di berkas terpisah `bab1_latar_belakang_pengembangan.md` (folder yang
sama), supaya kerangka di berkas ini tetap utuh sebagai acuan struktur. Daftar
pustaka BibTeX-nya ada di `daftar_pustaka_bab1.bib`.
