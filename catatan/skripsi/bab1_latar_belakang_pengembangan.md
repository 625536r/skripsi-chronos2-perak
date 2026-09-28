# Pengembangan 1.1 Latar Belakang — BAB 1 PENDAHULUAN

> Berkas terpisah dari `bab1.md` (kerangka). Isi di bawah ini adalah draf paragraf
> penuh untuk subbab 1.1, dilengkapi sitasi. Daftar pustaka lengkap dalam format
> BibTeX ada di `daftar_pustaka_bab1.bib` pada folder yang sama.
>
> Gaya sitasi: (Penulis, Tahun) — sesuaikan dengan format yang diwajibkan
> panduan skripsi program studi (APA/IEEE/lain) sebelum disalin ke naskah akhir.

## 1.1 Latar Belakang

Logam mulia telah lama menempati posisi ganda dalam sistem keuangan global:
sebagai komoditas dengan kegunaan industri sekaligus sebagai instrumen lindung
nilai (*safe haven*) yang diminati investor pada masa ketidakpastian ekonomi
(Mighri et al., 2022). Perak, secara khusus, memiliki karakter yang lebih rumit
dibandingkan emas — permintaannya berasal baik dari sektor investasi maupun
sektor industri (fotovoltaik, elektronik, dan manufaktur), sehingga harganya
cenderung lebih fluktuatif daripada emas meski keduanya sering dianggap
sekelompok (Kazak et al., 2026). Karena posisi ganda inilah,
memahami bagaimana harga Perak bergerak — baik sendiri maupun bersama
variabel makrofinansial lain — menjadi relevan baik untuk pengambil kebijakan
maupun pelaku pasar.

Salah satu keterkaitan yang paling sering diasumsikan dalam literatur finansial
adalah hubungan antara harga Perak, harga Emas, dan Dollar Index. Rasio
emas-perak dipandang sebagai relasi jangka panjang yang mapan, sementara Dollar
Index umumnya diasumsikan bergerak berlawanan arah dengan harga logam mulia
karena keduanya sama-sama mencerminkan preferensi terhadap aset berdenominasi
dolar Amerika Serikat. Namun demikian, sejumlah studi terbaru menunjukkan bahwa
pola keterkaitan antar-logam-mulia dan variabel makrofinansial ini tidak
sesederhana anggapan umum, dan cenderung berubah tergantung metode serta rezim
pasar yang diteliti. Sephton (2022) menemukan bahwa peran logam mulia sebagai
lindung nilai inflasi di sejumlah negara Afrika bergantung pada rezim
ekonometrika yang berlaku, bukan hubungan yang berlaku universal. Erdogan et al.
(2022), memakai uji kausalitas Granger nonparametrik pada seluruh distribusi,
menemukan bahwa arah keterkaitan antara logam mulia dan aset energi bersih
berbeda-beda pada bagian distribusi yang berbeda. Mighri et al. (2022) memakai
pendekatan kausalitas berbasis kuantil dan menemukan bahwa kausalitas Granger
antara indeks saham AS dan harga logam mulia bersifat *quantile-dependent* serta
berbeda-beda antar jenis logam. Kazak et al. (2026) menunjukkan bahwa hubungan
logam mulia dengan sektor riil (manufaktur) bersifat dinamis dan berubah
sepanjang waktu ketika diuji dengan analisis koherensi wavelet. Pola yang
konsisten muncul dari studi-studi ini: setiap studi umumnya hanya menerapkan
satu atau dua teknik uji (kointegrasi ambang, kausalitas dalam distribusi,
kausalitas berbasis kuantil, atau koherensi wavelet) pada satu sudut pandang
tertentu, bukan rangkaian uji bertahap yang saling memverifikasi mulai dari
stasioneritas hingga ketergantungan non-linear. Akibatnya, kesimpulan tentang
apakah Perak, Emas, dan Dollar Index benar-benar bergerak bersama, pada lag
berapa, dan ke arah mana presedensi prediktifnya berjalan, masih perlu diuji
secara lebih menyeluruh dan hati-hati agar tidak jatuh pada kesimpulan yang
keliru akibat regresi lancung pada deret yang tidak stasioner — sebuah risiko
metodologis yang secara eksplisit menjadi motivasi penerapan prosedur
Toda-Yamamoto pada pasangan variabel finansial lain, seperti keterkaitan
Bitcoin dengan indikator moneter dan ketidakpastian kebijakan ekonomi
(Sarker & Wang, 2022).

Di sisi peramalan, pendekatan konvensional untuk deret harga finansial memiliki
sejumlah keterbatasan yang mendasar. Model statistik klasik seperti ARIMA,
GARCH, atau VAR umumnya harus diestimasi ulang untuk tiap deret waktu secara
terpisah, dan sering bertumpu pada asumsi tertentu — misalnya stasioneritas
volatilitas pada model GARCH — yang pada praktiknya kerap dilanggar oleh data
finansial nyata (de Vilmarest & Werge, 2025). Pendekatan pembelajaran mesin yang
lebih modern pun tidak otomatis mengatasi masalah ini: banyak model masih gagal
mengestimasi ketidakpastian prediksinya sendiri secara memadai dan pada
dasarnya hanya menghasilkan satu angka prediksi titik tanpa gambaran seberapa
jauh nilai aktual dapat menyimpang dari angka tersebut (Renkema et al., 2024).
Padahal, dalam pengambilan keputusan finansial yang sarat ketidakpastian,
peramalan probabilistik boleh dibilang menjadi satu-satunya pendekatan yang
benar-benar sesuai, karena mampu menyajikan distribusi kemungkinan hasil alih-alih
sekadar titik ramalan tunggal (Smyl et al., 2026). Kebutuhan akan peramalan probabilistik
yang andal inilah yang mendorong berkembangnya berbagai metode kuantifikasi
ketidakpastian dalam satu dekade terakhir, mulai dari interval prediksi berbasis
statistik klasik hingga pendekatan pembelajaran mendalam (Sakib et al., 2025).

Perkembangan terbaru yang menjawab sebagian dari keterbatasan ini adalah
munculnya *time series foundation model* — model yang dilatih sekali pada
korpus deret waktu berskala besar dan lintas domain, kemudian dapat langsung
dipakai pada deret waktu baru tanpa perlu dilatih ulang dari awal (*zero-shot*),
atau disesuaikan lagi dengan sedikit data tambahan (*fine-tuned*). Chronos,
salah satu foundation model deret waktu pertama yang mendapat perhatian luas,
memperlakukan nilai deret waktu sebagai token dan melatih arsitektur model
bahasa berbasis transformer di atasnya, serta secara native menghasilkan
keluaran probabilistik dalam bentuk kuantil, bukan sekadar titik (Ansari et al.,
2024). Penerusnya, Chronos-2, memperluas kemampuan ini lebih jauh dengan
mekanisme *group attention* yang memungkinkan model memanfaatkan berbagai deret
yang saling terkait — termasuk target dan variabel kovariat — sekaligus dalam
satu proses peramalan *zero-shot*, tanpa memerlukan pelatihan khusus per tugas
(Ansari et al., 2025). Sejak diperkenalkan, keluarga model semacam ini mulai
diuji pada berbagai domain di luar tolok ukur akademik aslinya. Pada peramalan
tinggi gelombang laut, misalnya, skema *zero-shot* dan *fine-tuned* dari Chronos
diperbandingkan langsung pada domain yang sama, dan keduanya terbukti kompetitif
pada horizon yang berbeda-beda (Zhai et al., 2025). Pola serupa juga tampak pada
domain lain: performa *zero-shot* sering kali sudah kompetitif tanpa pelatihan
khusus, sementara manfaat tambahan dari *fine-tuning* bervariasi tergantung
karakteristik data domainnya masing-masing (Meyer et al., 2025).

Variasi hasil inilah yang menyisakan celah penelitian yang relevan secara
praktis: sejauh mana *fine-tuning* pada sebuah foundation model besar benar-benar
memberi manfaat ketika data yang tersedia hanya berasal dari satu deret waktu
tunggal, bukan ratusan atau ribuan deret seperti skenario pelatihan dan
pengujian foundation model pada umumnya. Tan Jerome & Simon (2026), melalui
analisis titik impas (*break-even analysis*) pada 30 kumpulan data tolok ukur,
menunjukkan bahwa keputusan antara memakai foundation model secara *zero-shot*,
melakukan *fine-tuning*, atau tetap memakai metode klasik sangat bergantung pada
panjang deret dan kekuatan pola musimannya — dan bahwa *fine-tuning* dengan
*LoRA* bahkan dapat menurunkan performa pada deret yang pendek. Temuan serupa
tentang perilaku foundation model yang tidak selalu dapat diprediksi lebih dulu
juga muncul dari pengujian bias sistematis Chronos-2 pada berbagai pola sintetis
(Jander et al., 2026). Pertanyaan tentang kapan *fine-tuning* layak dilakukan
pada skenario deret tunggal ini belum banyak dijawab secara spesifik pada kasus
komoditas finansial harian seperti harga Perak.

Berdasarkan dua celah tersebut — pertama, keterkaitan Perak-Emas-Dollar Index
yang tampak berlainan hasil tergantung metode dan belum diuji dengan rangkaian
uji bertahap yang saling memvalidasi; kedua, ketidakjelasan manfaat *fine-tuning*
foundation model deret waktu pada skenario deret tunggal — penelitian ini
diarahkan untuk mengisi keduanya secara konkret pada kasus harga Perak, dengan
Emas dan Dollar Index sebagai variabel kovariat, memakai lima tahun data harian.
Kedua sisi penelitian ini — analisis hubungan antar-variabel dan peramalan
probabilistik — dikerjakan sebagai dua tujuan yang setara kedudukannya, dengan
metodologi yang berbeda dan tidak saling menggantikan satu sama lain.
