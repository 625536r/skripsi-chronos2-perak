# Peramalan Probabilistik Harga Perak dengan Chronos-2

Skripsi S1 Teknik Informatika —
*Implementasi Time Series Foundation Model Chronos-2 untuk Peramalan Probabilistik
Harga Perak dengan Variabel Kovariat Harga Emas dan Dollar Index.*

Penelitian ini membandingkan dua skema penggunaan Chronos-2 — **zero-shot** dan
**fine-tuned** — untuk meramalkan harga Perak (`SI=F`), dengan Emas (`GC=F`) dan
Dollar Index (`DX-Y.NYB`) sebagai kovariat *past-only*.

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
→ (4) peramalan → (5) evaluasi & tabel BAB IV → (6) visualisasi.
Tanpa fine-tuning, seluruh pipeline selesai dalam **±80 detik di CPU**.

### Modul mandiri

```bash
python -m src.data_collection
python -m src.preprocessing
python -m src.mutual_information
python -m src.baseline
python -m src.evaluation
python -m src.forecasting --schemes zeroshot --split val   # uji asap, tidak menyentuh data uji
python -m src.visualization --split test                   # gambar ulang tanpa jalankan model
```

### Fine-tuning: jalankan di GPU (Colab)

Fine-tuning **tidak layak dijalankan di CPU** (lihat catatan keterbatasan).
Alur kerja dua mesin:

1. Di Colab dengan GPU T4:
   ```bash
   !git clone <repo> && cd skripsi-chronos2-perak
   !pip install -r requirements.txt
   !python run_pipeline.py --skip-download --scheme finetuned
   ```
2. Unduh dua berkas hasilnya:
   - `results/forecasts/finetuned_H10.parquet`
   - `results/metrics/finetune_tuning.json`
3. Salin keduanya ke lokasi yang sama di mesin lokal, lalu:
   ```bash
   python run_pipeline.py --report-only
   ```
   Tabel BAB IV, `final_results.json`, dan seluruh gambar akan tersusun ulang
   lengkap dengan baris Fine-Tuned dan uji Diebold-Mariano utama — tanpa
   menjalankan model apa pun lagi.

Daftar origin jendela bersifat deterministik (dibentuk `build_origins` dari
panjang data), sehingga hasil dari Colab otomatis berpijak pada **himpunan
jendela yang persis sama** dengan hasil lokal. Tahap 5 memverifikasi ini dan
akan menolak melanjutkan bila origin antar skema berbeda.

---

## Ringkasan hasil

Protokol: rolling origin (expanding window), H = 10 hari perdagangan, stride = 1,
**181 jendela** pada periode uji 2025-11-11 s.d. 2026-08-03.
Data: 1257 hari perdagangan (latih 879 / validasi 188 / uji 190).

### Mutual information (data latih saja, keputusan D6)

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
| Chronos-2 Fine-Tuned | _belum dijalankan_ | | | | | | | |

Uji Diebold-Mariano zero-shot vs naive (rugi = galat absolut, HAC Newey-West
lag 9): statistik 1,269, **p = 0,205 — tidak signifikan**.

**Temuan sementara.** Chronos-2 zero-shot **tidak terbukti berbeda** dari naive
persistence pada periode uji. Angka MAE-nya 6,1% lebih tinggi, tetapi selisih
itu tidak melewati ambang signifikansi, sehingga kesimpulan yang sah adalah
"tidak terdeteksi perbedaan", bukan "Chronos-2 lebih buruk". Hasil ini wajar
untuk harga logam mulia harian yang mendekati *martingale*. Pola per horizon
konsisten: kedua skema memburuk hampir sebanding akar horizon (MAE h=1 sebesar
2,50 vs 2,88; h=10 sebesar 8,01 vs 8,47).

Dari sisi kalibrasi, Chronos-2 menghasilkan interval **lebih sempit** daripada
baseline (Width80 14,88 vs 19,67) tetapi dengan cakupan yang meleset lebih jauh
(0,680 vs 0,768 terhadap target 0,80) — model terlalu percaya diri pada periode
uji yang bergejolak ini. *Quantile crossing rate* nol pada kedua skema.

---

## CATATAN PENTING — fine-tuning pada satu deret waktu

Dokumentasi AutoGluon menyarankan fine-tuning dilakukan ketika tersedia
**lebih dari 100 seri waktu**. Penelitian ini hanya memiliki **1 seri waktu**
(879 titik data latih).

Karena itu, **fine-tuning berpotensi TIDAK meningkatkan akurasi** dibanding
zero-shot — bahkan mungkin menurunkannya karena model sebesar Chronos-2 sangat
mudah *overfit* pada data sekecil ini.

**Ini bukan bug.** Bila hasilnya memang demikian, itu adalah **temuan penelitian
yang sah** dan **wajib dilaporkan apa adanya**, bukan disembunyikan, bukan
diakali dengan memilih ulang hyperparameter memakai data uji, dan bukan alasan
untuk mengubah protokol evaluasi setelah melihat hasil. Justru inilah kontribusi
empiris yang dapat ditawarkan skripsi ini: bukti terukur tentang kapan
fine-tuning sebuah *time series foundation model* layak dilakukan dan kapan
tidak, pada kasus deret tunggal harga komoditas.

---

## Catatan keterbatasan

1. **Skema fine-tuned belum dijalankan.** Fine-tuning terukur ±12 detik per
   langkah gradien di CPU mesin ini (Chronos-2 120M parameter, tanpa CUDA),
   bahkan setelah `fine_tune_batch_size` dan `fine_tune_context_length`
   diturunkan agar sepadan dengan ukuran data. Grid pada config (3 lr × 3 steps
   × 2 mode = 18 kandidat, 10.200 langkah) berarti **lebih dari 30 jam**. Tahap
   ini dipindahkan ke GPU — lihat "Fine-tuning: jalankan di GPU" di atas.
   Seluruh baris Fine-Tuned pada tabel dan seluruh angka uji Diebold-Mariano
   utama (zero-shot vs fine-tuned) **masih kosong** sampai langkah itu selesai.

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
   pada H = 10 hari perdagangan. Sensitivitas H ∈ {5, 20} disediakan lewat
   `--horizon` tetapi dilaporkan terpisah.

6. **Ablasi oracle belum dijalankan.** Skenario `known_covariates` sebagai
   batas atas teoretis (`model.oracle_ablation.enabled`) masih `false`.

---

## Catatan implementasi yang sudah diverifikasi

Diverifikasi pada `autogluon.timeseries==1.6.1`, `chronos==2.3.1`, `torch==2.13.0+cpu`:

| Hal | Fakta |
|---|---|
| `fine_tune_mode` | Hanya menerima `"lora"` atau `"full"`. Nilai `"linear_probe"` **tidak didukung** dan menggagalkan `fit()`. |
| Kovariat (D2) | `known_covariates_names=[]`; `gold_close` dan `dxy_close` otomatis menjadi *past covariates*. Nilai masa depannya tidak pernah diteruskan ke `predict()`. |
| Indeks waktu | Kalender bursa tidak reguler (`freq=None`); tanggal asli dipetakan 1:1 ke indeks sintetis berfrekuensi tetap. Seluruh keluaran ke disk tetap memakai tanggal perdagangan asli. |
| Perangkat | CUDA tidak tersedia di mesin ini; seluruh eksekusi lokal berjalan di CPU. |

---

## Keluaran

| Berkas | Isi |
|---|---|
| `results/metrics/comparison_table.md` | Tabel siap salin ke BAB IV (akurasi, kalibrasi, uji DM) |
| `results/metrics/final_results.json` | Seluruh metrik: keseluruhan, per horizon, uji DM, protokol |
| `results/metrics/finetune_tuning.json` | Tabel lengkap tuning hyperparameter (termasuk kandidat yang kalah) |
| `results/forecasts/{skema}_H{H}.parquet` | Ramalan mentah tiap skema — evaluasi dapat diulang tanpa menjalankan model |
| `results/figures/01..05_*.png` | Lima gambar laporan, 300 dpi, berlabel bahasa Indonesia |
| `results/environment.json` | Versi python dan seluruh library |
| `logs/*.log` | Log tiap modul dan tiap tahap pipeline |
