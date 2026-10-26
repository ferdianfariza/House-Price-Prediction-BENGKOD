# EARLY BIRD
## BENGKEL KODING DATA SCIENCE
---

## A. Latar Belakang

Harga rumah merupakan salah satu informasi paling menentukan dalam keputusan jual beli properti. Penjual membutuhkan patokan harga yang wajar agar listing tidak terlalu mahal maupun terlalu murah, sedangkan pembeli dan agen membutuhkan dasar untuk membandingkan penawaran antar lokasi. Memahami faktor-faktor yang membentuk harga, mulai dari luas tanah dan bangunan hingga lokasi, menjadi kunci bagi pelaku pasar properti dalam mengambil keputusan yang lebih terukur.

Dataset **"Listing Rumah (Raw-Final-Rumah)"** berisi data listing rumah di wilayah Depok dan sekitarnya yang diambil dari listing properti daring melalui proses *scraping*. Data ini mencakup harga penawaran, jumlah kamar tidur, kamar mandi, garasi, luas tanah, luas bangunan, dan lokasi. Karena masih berupa data mentah, kualitas data menjadi tantangan utama sebelum model dapat dibangun. Tantangan utamanya adalah memprediksi harga rumah (regresi) berdasarkan karakteristik fisik dan lokasi listing.

Anda berperan sebagai Data Scientist yang ditugaskan untuk menganalisis dataset. Proyek ini akan menguji kompetensi Anda dalam menerapkan siklus hidup Data Science secara utuh, mulai dari Eksplorasi Data (EDA), Pemodelan Langsung (*Direct Modeling*), Pembersihan dan Pra Pemrosesan Data, hingga Pemodelan Ulang, Perbandingan Hasil, dan Evaluasi menggunakan model regresi, serta optimasi melalui Validasi dan *Hyperparameter Tuning*.

## B. Dataset

1. Link: [Dataset](https://drive.google.com/file/d/1T_wjbyeNGFib77hi4a61ktfTmJLLk0UQ/view)
2. Deskripsi dataset:

| NO | Feature | Type | Keterangan Singkat |
|----|---------|------|--------------------|
| 1 | Harga | object | Harga penawaran listing, ditulis dalam bentuk teks. **(Target)** |
| 2 | Kamar Tidur | object | Jumlah kamar tidur. |
| 3 | Kamar Mandi | object | Jumlah kamar mandi. |
| 4 | Garasi | object | Informasi garasi pada listing. |
| 5 | Luas Tanah | object | Luas tanah (satuan m²). |
| 6 | Luas Bangunan | object | Luas bangunan (satuan m²). |
| 7 | Kecamatan | object | Lokasi listing. |
| 8 | Page | int64 | Kolom pendukung dari proses pengambilan data; tentukan sendiri relevansinya. |

**Petunjuk Dataset:**

1. Jumlah data sekitar 40.000 baris. Sebagian besar kolom tersimpan sebagai teks bersatuan (misalnya "2,5 Miliar" dan "LT : 438 m²") sehingga perlu diparsing menjadi angka. Format penulisan harga tidak hanya seperti contoh tersebut; periksa seluruh variasi satuan dan keterangan tambahan yang muncul.
2. Isi kolom tidak selalu sesuai dengan namanya pada semua baris. Periksa pola isi tiap kolom sebelum menentukan penanganan.
3. Terdapat *missing value* dan data duplikat. Telusuri penyebabnya sebelum menentukan penanganan; duplikat tidak selalu berupa baris yang identik pada semua kolom.
4. Terdapat nilai yang tampak tidak wajar pada harga dan luas. Bedakan nilai ekstrem yang valid dari kesalahan input.
5. Tidak semua *listing* bermakna sama dan tidak semua kolom relevan untuk pemodelan.
6. Isi kolom Kecamatan belum tentu berada pada tingkat wilayah yang sama.
7. Distribusi harga diperkirakan tidak simetris dan jumlah data antar lokasi tidak merata.

## C. Poin Penilaian

### 1. EDA

Melakukan eksplorasi data secara menyeluruh yang mencakup identifikasi karakteristik dan kualitas data. Berdasarkan hasil eksplorasi tersebut, lakukan visualisasi data yang relevan serta jelaskan kebutuhan *preprocessing* atau transformasi data untuk tahap selanjutnya.

Mahasiswa wajib menyertakan minimal 3 insight utama yang berkaitan langsung dengan variabel target (*Harga*). Insight yang dimaksud tidak hanya berupa deskripsi hasil visualisasi, tetapi juga harus menunjukkan pemahaman terhadap pola yang ditemukan. Setiap insight perlu dihubungkan dengan keputusan yang diambil pada tahap berikutnya, seperti alasan melakukan *preprocessing* tertentu atau pemilihan model.

**Luaran:** tabel temuan kualitas data (masalah, jumlah baris, dan kemungkinan penyebab), visualisasi terpilih, dan minimal 3 insight yang dihubungkan dengan keputusan tahap berikutnya.

### 2. Direct Modeling

Sebelum *preprocessing*, bangunlah minimal 5 model regresi (termasuk model *ensemble*) secara langsung. Pemilihan model bersifat bebas, namun setiap pilihan harus didasari oleh alasan yang logis dan relevan dengan karakteristik dataset. Hasil dari tahap ini akan digunakan sebagai kontrol untuk membuktikan efektivitas teknik *preprocessing* dan *feature engineering* yang akan diterapkan pada tahap berikutnya. Alasan pemilihan model juga harus ditulis dalam *markdown* pada notebook.

### 3. Modeling Dengan Preprocessing

Dengan model yang sama seperti direct modeling, lakukan preprocessing untuk menghasilkan prediksi dengan data yang sudah bersih.

### 4. Feature Selection

Lakukan proses feature selection untuk memilih fitur yang paling relevan terhadap target menggunakan metode bebas, seperti korelasi, atau metode lainnya. Gunakan fitur hasil seleksi untuk melatih model.

### 5. Hyperparameter Tuning

Optimalkan model yang diperlukan dengan *hyperparameter tuning* (misalnya *Grid Search* atau *Randomized Search*) menggunakan *cross-validation* pada data latih saja. Setelahnya, analisis model mana yang memiliki performa terbaik berdasarkan MAE, MAPE, dan R² pada *test set* bersih, yang nantinya akan digunakan di *deployment*.

### 6. Deployment

Deployment dilakukan dengan mengimplementasikan model terbaik ke dalam aplikasi berbasis Streamlit Cloud agar dapat digunakan secara publik. Proses ini dimulai dari menyiapkan model evaluasi dan hyperparameter tuning, kemudian menyimpannya dalam format seperti `.pkl` atau `.joblib` agar dapat dimuat dalam aplikasi. Selanjutnya, dibuat aplikasi Streamlit yang memungkinkan pengguna memasukan fitur melalui form, menjalankan proses prediksi, serta menampilkan hasil secara langsung. Aplikasi juga dilengkapi dengan elemen pendukung seperti penjelasan fitur dan visualisasi agar lebih mudah dipahami.

Sebelum dipublikasikan, aplikasi diuji terlebih dahulu secara lokal untuk memastikan seluruh fungsinya berjalan dengan baik. Setelah itu, disiapkan file pendukung seperti `requirements.txt` agar dependensi dapat terinstal dengan benar. Proyek kemudian diunggah ke GitHub sebagai sumber kode, dan tahap akhir dilakukan deployment ke Streamlit Cloud hingga aplikasi dapat diakses secara publik.

### 7. Presentasi

Presentasi dilaksanakan secara offline sesuai dengan jadwal sesi kelas yang telah ditentukan. Setiap mahasiswa diberikan waktu 15 menit untuk mendemonstrasikan hasil deployment dan menjelaskan kode program yang telah dibuat. Perlu diingat bahwa kesempatan presentasi hanya satu kali tanpa ada revisi. Mahasiswa yang dinyatakan lolos lebih awal berhak mendapatkan kompensasi **bebas perkuliahan (*free class*)** untuk pertemuan-pertemuan berikutnya. Dengan demikian, semakin cepat Anda mengambil sesi presentasi dan dinyatakan lolos, semakin banyak waktu bebas yang akan didapatkan.

**Jadwal Presentasi**

| DS-01 | DS-02 | DS-03 | DS-04 |
|-------|-------|-------|-------|
| Koordinasi Dengan PJ Kelas | Koordinasi Dengan PJ Kelas | Koordinasi Dengan PJ Kelas | Koordinasi Dengan PJ Kelas |

## D. Pengumpulan

a. Pengumpulan dilakukan melalui website Bengkel Koding.
b. Pekerjaan wajib dikumpulkan H-1 **sebelum sesi presentasi**.
c. Format pengumpulan berupa PDF yang memuat:
   - Nama.
   - NIM.
   - Link repository GitHub berisi source code.
   - Link aplikasi Streamlit yang telah berhasil di-deploy.
   - Link Notebook Collab

## E. Kriteria Penilaian

a. Kelengkapan pengumpulan (10%):
   - Link repository GitHub yang berisi source code project (`.ipynb`)
   - Link aplikasi Streamlit yang telah berhasil di-deploy.
   - Link Notebook Collab

b. Penyampaian **presentasi** (40%).

c. Pemahaman dan Pertanyaan (50%)