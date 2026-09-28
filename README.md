# Peramalan Probabilistik Harga Perak dengan Chronos-2

Skripsi S1 Teknik Informatika —
*Implementasi Time Series Foundation Model Chronos-2 untuk Peramalan Probabilistik
Harga Perak dengan Variabel Kovariat Harga Emas dan Dollar Index.*

Penelitian ini punya dua tujuan yang setara kedudukannya, dengan metodologi berbeda:

1. **Analisis hubungan antar-variabel** — menyelidiki keterkaitan Perak, Emas, dan
   Dollar Index: apakah bergerak bersama, pada lag berapa, dan ke arah mana presedensi
   prediktifnya (`src/relationship_analysis.py`).
2. **Peramalan probabilistik** — membandingkan dua skema penggunaan Chronos-2 —
   **zero-shot** dan **fine-tuned** — untuk meramalkan harga Perak (`SI=F`), dengan Emas
   (`GC=F`) dan Dollar Index (`DX-Y.NYB`) sebagai kovariat *past-only* (`run_pipeline.py`).

> Berkas ini menjelaskan **cara menjalankan** dan **ringkasan hasil**.
> Untuk memahami **konsep, keputusan metodologis, dan tafsir hasilnya**, baca
> [`PANDUAN_PROJECT.md`](PANDUAN_PROJECT.md).

---

## Cara menjalankan

### Persiapan

```bash
pip install -r requirements.txt
python -m src.utils          # mencatat versi seluruh library ke results/environment.json
```

### Pipeline lengkap

```bash
python run_pipeline.py                             # tahap 1-6, seluruh skema
python run_pipeline.py --skip-download             # pakai data/raw/ yang sudah ada
python run_pipeline.py --horizon 5                 # horizon lain
python run_pipeline.py --scheme zeroshot naive     # sebagian skema saja
python run_pipeline.py --report-only               # susun ulang tabel + gambar saja
```

Tahapan: (1) pengumpulan data → (2) preprocessing & split → (3) mutual information
→ (4) peramalan → (5) evaluasi & tabel hasil → (6) visualisasi. Keenamnya melayani
**tujuan kedua** (peramalan).
Tanpa fine-tuning, seluruh pipeline selesai dalam **±80 detik di CPU**.
Dengan fine-tuning (skema `finetuned`, grid 18 kandidat) di GPU lokal
**RTX 3050 6GB Laptop GPU**, seluruh pipeline selesai dalam **±22 menit**.

### Analisis hubungan antar-variabel (tujuan pertama)

Modul terpisah, di luar `run_pipeline.py` di atas — hanya bergantung pada
`data/processed/train.csv` (hasil tahap 2) dan, sebagai pelengkap, `mutual_information.json`
(hasil tahap 3):

```bash
python -m src.relationship_analysis
```

Menjalankan tujuh langkah berurutan pada data latih saja (stasioneritas → ACF/PACF →
cross-correlation → kointegrasi → pemilihan lag VAR → diagnostik residual → Granger
causality dengan prosedur Toda-Yamamoto), lalu mengutip Mutual Information sebagai
pelengkap non-linear. Selesai dalam hitungan detik, tidak perlu GPU. Ringkasan hasilnya
di bawah; detail lengkap di `results/metrics/relationship_summary.md`.

### Modul mandiri

```bash
python -m src.data_collection
python -m src.preprocessing
python -m src.mutual_information
python -m src.relationship_analysis                         # tujuan pertama, lihat di atas
python -m src.baseline
python -m src.evaluation
python -m src.forecasting --schemes zeroshot --split val   # uji asap, tidak menyentuh data uji
python -m src.visualization --split test                   # gambar ulang tanpa jalankan model
python -m src.sensitivity --reuse-finetune-params          # horizon lain, ablasi kovariat, oracle
python -m src.sensitivity --summary-only                   # susun ulang ringkasan sensitivitas saja
```

### Fine-tuning: jalankan di GPU lokal

Fine-tuning **tidak layak dijalankan di CPU** (lihat catatan keterbatasan) —
grid 18 kandidat (10.200 langkah gradien total) diekstrapolasi >30 jam di CPU.
Mesin pengembangan sekarang punya GPU lokal (**NVIDIA RTX 3050 6GB Laptop
GPU**, torch `cu126`), sehingga seluruh pipeline — termasuk skema
`finetuned` — cukup dijalankan langsung di satu mesin yang sama:

```bash
python run_pipeline.py --scheme finetuned    # atau tanpa --scheme untuk semua skema
```

Grid 18 kandidat selesai dalam **±21 menit** di GPU ini (tercatat pada
`results/metrics/finetune_tuning.json`, field `total_seconds`); seluruh
pipeline (tahap 1-6, seluruh skema) selesai dalam **±22 menit**. Bila GPU
tidak tersedia, `torch`/AutoGluon otomatis jatuh ke CPU — pada kondisi itu
jalankan skema lain dulu (`--scheme zeroshot naive`) dan pertimbangkan
menurunkan grid pada `config.yaml` bagian `finetune.grid`.

Riwayat: sebelum GPU lokal tersedia, tahap fine-tuning sempat dijalankan di
Colab (GPU Tesla T4) sebagai mesin kedua — jejaknya masih ada di
`results/environment_colab.json` untuk keperluan arsip, tapi alur itu
**sudah tidak dipakai**. Daftar origin jendela bersifat deterministik
(dibentuk `build_origins` dari panjang data), sehingga hasil dari mesin
manapun tetap berpijak pada himpunan jendela yang sama; tahap 5 memverifikasi
ini dan akan menolak melanjutkan bila origin antar skema berbeda.

---

## Ringkasan analisis hubungan (tujuan pertama)

Dihitung `src/relationship_analysis.py`, data latih saja. Detail lengkap di
`results/metrics/relationship_summary.md`.

**Stasioneritas.** Perak, Emas, dan DXY **tidak stasioner pada level**, **stasioner pada
log-return** (ADF dan KPSS sepakat untuk ketiganya), ordo integrasi d = 1.

**Cross-correlation (log-return, lag −10 s.d. +10).** Korelasi memuncak tajam di **lag 0**
untuk kedua pasangan — Perak–Emas **0,778**, Perak–DXY **−0,406** — dan tidak signifikan
di lag manapun selain itu.

**Kointegrasi.** Engle-Granger tidak menemukan kointegrasi pada pasangan manapun
(α = 0,05; Perak–Emas paling dekat, p = 0,057). Johansen ambigu: rank trace test = 1,
rank max-eigenvalue test = 0.

**Presedensi prediktif (Granger causality, prosedur Toda-Yamamoto, lag p = 3, HAC):**

| Pasangan | Statistik Wald | df | p-value | Signifikan (α=0,05) |
|---|---:|---:|---:|---|
| Emas Granger-cause Perak | 3,924 | 3 | 0,270 | tidak |
| DXY Granger-cause Perak | 1,962 | 3 | 0,580 | tidak |
| Perak Granger-cause Emas | 4,440 | 3 | 0,218 | tidak |
| Perak Granger-cause DXY | 0,525 | 3 | 0,913 | tidak |

Tidak ada satu pun arah yang signifikan — konsisten dengan Mutual Information berlag yang
juga runtuh ke ≈ 0 pada lag ≥ 1. **Simpulan:** ketiga seri terbukti berkaitan kuat, tapi
**sezaman**, bukan salah satu mendahului yang lain. Empat jalur independen (CCF,
kointegrasi, Toda-Yamamoto, MI) menunjuk kesimpulan yang sama. Tafsir lengkap, termasuk
kenapa dipilih prosedur Toda-Yamamoto, ada di `PANDUAN_PROJECT.md` bagian 6.

---

## Ringkasan hasil peramalan (tujuan kedua)

Protokol: rolling origin (expanding window), H = 10 hari perdagangan, stride = 1,
**181 jendela** pada periode uji 2025-11-11 s.d. 2026-08-03.
Data: 1257 hari perdagangan (latih 879 / validasi 188 / uji 190).

### Mutual information (data latih saja)

| Varian | Emas | DXY |
|---|---|---|
| Level harga | 1,348 nat | 0,775 nat |
| Log-return | 0,415 nat | 0,090 nat |

MI level 3,2–8,6 kali lebih besar daripada MI log-return, persis seperti yang
diperkirakan: pada level, estimator k-NN sebenarnya mengukur kesamaan lintasan
tren antar proses non-stasioner, bukan keterkaitan pergerakan harian. Uji
permutasi pada log-return tetap menolak kemandirian untuk kedua kovariat
(p ≤ 0,001). MI log-return runtuh menjadi ≤ 0,005 nat begitu lag ≥ 1, sehingga
kovariat bersifat **sezaman dan bukan penunjuk arah** — manfaatnya bagi
Chronos-2 terletak pada pengayaan konteks *past-only*, bukan pada kemampuan
meramal dari kovariat saja.

### Perbandingan skema (periode uji)

| Skema | MAE | RMSE | MAPE | MASE | CRPS | Cov80 | Cov95 | Width80 |
|---|---|---|---|---|---|---|---|---|
| Naive Persistence | 5,681 | 8,295 | 7,51% | 1,000 | 3,700 | 0,768 | 0,915 | 19,67 |
| Chronos-2 Zero-Shot | 6,030 | 9,104 | 7,98% | 1,062 | 3,877 | 0,680 | 0,909 | 14,88 |
| Chronos-2 Fine-Tuned | 6,058 | 8,986 | 8,01% | 1,066 | 3,958 | 0,663 | 0,875 | 14,81 |

Uji Diebold-Mariano (rugi = galat absolut, HAC Newey-West lag 9):

| Perbandingan | Statistik DM | p-value | Signifikan (α = 0,05) |
|---|---|---|---|
| Zero-Shot vs Fine-Tuned **(uji utama)** | −0,202 | 0,8401 | tidak |
| Zero-Shot vs Naive | 1,269 | 0,2045 | tidak |
| Fine-Tuned vs Naive | 1,811 | 0,0702 | tidak |

**Temuan 1 — fine-tuning tidak mengubah apa pun.** Selisih MAE zero-shot dan
fine-tuned hanya 0,028 USD dengan p = 0,8401: praktis tidak ada perbedaan sama
sekali. Bukti pendukungnya ada di dalam proses tuning itu sendiri
(`finetune_tuning.json`): dari 18 kandidat, yang menang adalah **LoRA dengan
langkah paling sedikit** (lr 1e-4, 200 langkah, WQL 0,009859), sedangkan yang
paling buruk adalah **full fine-tuning dengan langkah terbanyak** (lr 1e-4,
1.000 langkah, WQL 0,026091 — 2,6 kali lebih jelek). Pola itu persis tanda
*overfitting*: makin banyak model diubah pada data sekecil ini, makin rusak
hasilnya. Lihat catatan "fine-tuning pada satu deret waktu" di bawah.

**Temuan 2 — Chronos-2 tidak terbukti berbeda dari naive persistence.** MAE
zero-shot 6,2% lebih tinggi, tetapi p = 0,205, sehingga kesimpulan yang sah
adalah "tidak terdeteksi perbedaan", bukan "Chronos-2 lebih buruk". Hasil ini
wajar untuk harga logam mulia harian yang mendekati *martingale*. Pola per
horizon konsisten: seluruh skema memburuk hampir sebanding akar horizon (MAE
h=1 sebesar 2,50 / 2,88 / 2,90; h=10 sebesar 8,01 / 8,47 / 8,45).

**Temuan 3 — interval Chronos-2 terlalu percaya diri.** Kedua skema Chronos-2
menghasilkan interval **lebih sempit** daripada baseline (Width80 14,88 dan
14,81 vs 19,67) tetapi dengan cakupan yang meleset lebih jauh (0,680 dan 0,663
vs 0,768 terhadap target 0,80). Yang lebih tajam terlihat per horizon: cakupan
80% zero-shot nyaris tepat sasaran pada h=1 (0,812) lalu runtuh pada h=10
(0,630), sedangkan baseline lebih stabil (0,801 → 0,735). Model **tidak
melebarkan ketidakpastiannya secepat yang seharusnya** seiring horizon
memanjang. *Quantile crossing rate* nol pada ketiga skema.

### Analisis sensitivitas

Dilaporkan **terpisah** di `results/metrics/sensitivity_summary.md` agar tidak
tercampur dengan hasil utama.

| Analisis | Hasil ringkas |
|---|---|
| **Horizon** H ∈ {5, 10, 20} | Kesimpulan bertahan. MASE zero-shot 1,077 / 1,062 / 1,051; uji DM zero-shot vs fine-tuned tidak signifikan di ketiga horizon (p = 0,835 / 0,840 / 0,941). |
| **Ablasi kovariat** | Hanya Perak 6,199 → +Emas 6,224 → +Emas+DXY 6,030 (MAE). Uji DM utama **p = 0,332 — kovariat tidak terbukti membantu**, konsisten dengan MI log-return yang runtuh pada lag ≥ 1. |
| **Oracle / ex-post** (TIDAK REALISTIS) | Bila nilai aktual kovariat pada horizon ramalan diberikan ke model, MAE turun 6,030 → 5,015 (p = 0,0006). Ini **bukan** potensi perbaikan yang dapat dicapai — ia mengukur seberapa besar keuntungan tak sah yang muncul bila pemisahan *past-only* dilanggar. |

---

## CATATAN PENTING — fine-tuning pada satu deret waktu

Dokumentasi AutoGluon menyarankan fine-tuning dilakukan ketika tersedia
**lebih dari 100 seri waktu**. Penelitian ini hanya memiliki **1 seri waktu**
(879 titik data latih).

Dugaan awalnya, **fine-tuning berpotensi tidak meningkatkan akurasi** dibanding
zero-shot karena model sebesar Chronos-2 sangat mudah *overfit* pada data
sekecil ini. **Hasilnya memang demikian** (p = 0,8401), dan tanda *overfitting*
terlihat langsung pada peringkat kandidat tuning.

**Ini bukan bug.** Ini adalah **temuan penelitian yang sah** dan **wajib
dilaporkan apa adanya**, bukan disembunyikan, bukan
diakali dengan memilih ulang hyperparameter memakai data uji, dan bukan alasan
untuk mengubah protokol evaluasi setelah melihat hasil. Justru inilah kontribusi
empiris yang dapat ditawarkan skripsi ini: bukti terukur tentang kapan
fine-tuning sebuah *time series foundation model* layak dilakukan dan kapan
tidak, pada kasus deret tunggal harga komoditas.

---

## Catatan keterbatasan

1. **Fine-tuning memerlukan GPU.** Di CPU, fine-tuning terukur ±12 detik per
   langkah gradien (Chronos-2 120M parameter, tanpa CUDA), bahkan setelah
   `fine_tune_batch_size` dan `fine_tune_context_length` diturunkan agar
   sepadan dengan ukuran data. Grid pada config (3 lr × 3 steps × 2 mode =
   18 kandidat, 10.200 langkah) berarti lebih dari 30 jam di CPU — tidak
   layak. Angka Fine-Tuned yang dilaporkan di sini dihasilkan di **GPU lokal
   (NVIDIA RTX 3050 6GB Laptop GPU)**, selesai dalam **±21 menit** — lihat
   "Fine-tuning: jalankan di GPU lokal" di atas. Seluruh skema (Zero-Shot,
   Fine-Tuned, Naive) sekarang dihasilkan pada mesin dan sesi yang sama;
   `results/environment.json` mencatat lingkungan aktualnya. Sempat ada
   percobaan awal memakai GPU Tesla T4 di Colab sebelum GPU lokal tersedia —
   jejaknya masih tersimpan di `results/environment_colab.json` sebagai
   arsip, tapi tidak lagi dipakai untuk menghasilkan angka pada laporan ini.

2. **Periode uji jatuh pada rezim pasar ekstrem.** Harga perak naik sekitar
   lima kali lipat dari basis 2021 dan memuncak tajam di dalam jendela uji. MAE
   pada periode uji (5,68 untuk baseline) lebih dari empat kali MAE pada periode
   validasi (1,23). Angka absolut pada tabel karena itu **tidak dapat
   dibandingkan** dengan penelitian yang memakai periode lain, dan kesimpulan
   tentang kalibrasi interval terikat pada rezim bergejolak ini.

3. **Kovariat bersifat sezaman, bukan lead.** MI log-return runtuh ke ≈ 0 pada
   lag ≥ 1, sehingga secara teoretis kovariat tidak menyimpan informasi
   prediktif beberapa hari ke depan. Ini membatasi seberapa besar keuntungan
   yang mungkin diperoleh dari skema kovariat *past-only* mana pun.

4. **Kalender bursa dipetakan ke indeks sintetis.** Kalender perdagangan tidak
   berfrekuensi tetap sehingga ditolak AutoGluon. Tanggal asli dipetakan 1:1 ke
   indeks berfrekuensi tetap, sehingga 1 langkah model = 1 hari perdagangan.
   Konsekuensinya jarak kalender yang sesungguhnya (akhir pekan, hari libur)
   tidak dilihat model. Ini aman untuk Chronos-2 yang tidak memakai timestamp
   sebagai fitur, tetapi tidak otomatis berlaku untuk model lain.

5. **Satu aset, satu horizon utama.** Seluruh kesimpulan berlaku untuk perak
   pada H = 10 hari perdagangan. Sensitivitas H ∈ {5, 20} sudah dijalankan dan
   mendukung kesimpulan yang sama, tetapi dilaporkan terpisah.

6. **Fine-tuned pada horizon sensitivitas memakai ulang hyperparameter H = 10**,
   tanpa tuning grid ulang per horizon, demi kelayakan komputasi. Ini
   penyimpangan dari protokol pemilihan hyperparameter yang dicatat terbuka di
   `sensitivity_summary.md` dan `sensitivity_horizon.json`.

7. **Skenario oracle bersifat ex-post dan tidak realistis.** Ia dijalankan lewat
   `python -m src.sensitivity` (bukan lewat `model.oracle_ablation.enabled`,
   yang tetap `false`) dan seluruh keluarannya berlabel "EX-POST / TIDAK
   REALISTIS". Angkanya tidak boleh dinarasikan sebagai performa yang dapat
   dicapai model.

---

## Catatan implementasi yang sudah diverifikasi

Diverifikasi pada `autogluon.timeseries==1.6.1`, `chronos==2.3.1`, `torch==2.13.0+cu126`:

| Hal | Fakta |
|---|---|
| `fine_tune_mode` | Hanya menerima `"lora"` atau `"full"`. Nilai `"linear_probe"` **tidak didukung** dan menggagalkan `fit()`. |
| Kovariat (past-only) | `known_covariates_names=[]`; `gold_close` dan `dxy_close` otomatis menjadi *past covariates*. Nilai masa depannya tidak pernah diteruskan ke `predict()`. |
| Indeks waktu | Kalender bursa tidak reguler (`freq=None`); tanggal asli dipetakan 1:1 ke indeks sintetis berfrekuensi tetap. Seluruh keluaran ke disk tetap memakai tanggal perdagangan asli. |
| Perangkat | CUDA tersedia di mesin lokal (**NVIDIA RTX 3050 6GB Laptop GPU**); fine-tuning dan peramalan berjalan di GPU. |

---

## Keluaran

| Berkas | Isi |
|---|---|
| `results/metrics/relationship_analysis.json` | Hasil lengkap tujuan pertama: stasioneritas, kointegrasi, lag VAR, diagnostik residual, Granger Toda-Yamamoto |
| `results/metrics/relationship_summary.md` | Lima tabel siap salin (ordo integrasi, kointegrasi, lag VAR, presedensi prediktif vs MI, diagnostik residual) |
| `results/figures/relationship/acf_pacf.png`, `ccf_perak_emas.png`, `ccf_perak_dxy.png` | Tiga gambar tujuan pertama |
| `results/metrics/comparison_table.md` | Tabel siap salin ke laporan (akurasi, kalibrasi, uji DM) |
| `results/metrics/final_results.json` | Seluruh metrik: keseluruhan, per horizon, uji DM, protokol |
| `results/metrics/finetune_tuning.json` | Tabel lengkap tuning hyperparameter (termasuk kandidat yang kalah) |
| `results/forecasts/{skema}_H{H}.parquet` | Ramalan mentah tiap skema — evaluasi dapat diulang tanpa menjalankan model |
| `results/figures/evaluasi/01..05_*.png` | Lima gambar laporan, 300 dpi, berlabel bahasa Indonesia |
| `results/metrics/sensitivity_summary.md` | Ringkasan tiga analisis sensitivitas |
| `results/metrics/sensitivity_*.json` | Angka mentah tiap analisis sensitivitas |
| `results/environment.json` | Versi python, library, dan perangkat (GPU lokal) yang menghasilkan angka pada laporan ini |
| `results/environment_colab.json` | Arsip dari percobaan Colab sebelum GPU lokal tersedia — tidak lagi dipakai |
| `logs/*.log` | Log tiap modul dan tiap tahap pipeline |
