# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang berdiri sejak tahun 2000 dan telah membangun reputasi yang sangat baik dengan mencetak banyak lulusan berkualitas. Meskipun demikian, institusi ini menghadapi tantangan signifikan terkait tingginya angka mahasiswa yang tidak menyelesaikan masa studinya atau dropout.

### Permasalahan Bisnis

Permasalahan utama yang dihadapi oleh Jaya Jaya Institut adalah tingginya jumlah mahasiswa yang dropout. Kondisi ini membawa dampak negatif bagi institusi, antara lain:

- Merusak reputasi akademik institusi.
- Mempengaruhi stabilitas finansial karena pendapatan dari biaya kuliah menurun.
- Menurunkan efisiensi alokasi sumber daya seperti fasilitas dan tenaga pengajar.

Oleh karena itu, institusi berupaya untuk mengidentifikasi mahasiswa yang berisiko dropout sedini mungkin. Dengan deteksi dini, pihak institut dapat memberikan intervensi berupa bimbingan khusus untuk membantu mahasiswa tersebut agar dapat menyelesaikan pendidikannya.

### Cakupan Proyek

Proyek ini berfokus pada analisis data historis mahasiswa untuk menemukan pola yang berkontribusi terhadap status kelulusan dan dropout. Cakupan proyek ini meliputi:

- Analisis Data Eksploratif: Menganalisis dan memvisualisasikan data mahasiswa untuk menemukan faktor-faktor kunci yang membedakan mahasiswa Lulus, Dropout, dan yang masih aktif.
- Pembuatan Dasbor Bisnis: Menyajikan hasil analisis dalam sebuah dasbor interaktif untuk memudahkan pemangku kepentingan dalam memahami kondisi dan tren data.
- Pengembangan Model Machine Learning: Membangun model klasifikasi untuk memprediksi status masa depan seorang mahasiswa (Lulus atau Dropout).
- Pembuatan Prototipe: Mengembangkan aplikasi web sederhana sebagai prototipe dari sistem prediksi yang dapat digunakan oleh pihak administrasi institut.

### Persiapan

- Sumber Data: Data yang digunakan adalah data historis yang mencakup informasi demografis, jalur pendaftaran, latar belakang orang tua, status finansial, dan performa akademik pada dua semester pertama. Dataset dapat didownload di https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- Setup Environment: Untuk menjalankan prototipe sistem machine learning secara lokal, dibutuhkan beberapa library Python yang tercantum dalam file requirements.txt.

## Business Dashboard

Sebuah dasbor bisnis telah dibuat untuk memvisualisasikan data historis mahasiswa dan mengidentifikasi tren utama terkait status kelulusan, dropout, dan mahasiswa aktif (Enrolled). Dasbor ini menyajikan berbagai wawasan penting dari data.

Beberapa temuan kunci dari dasbor:

- Distribusi Status Mahasiswa: Dari total mahasiswa dalam dataset, 2.209 Lulus, 1.421 Dropout, dan 794 masih terdaftar sebagai mahasiswa aktif (Enrolled).
- Faktor Keuangan: Mahasiswa yang membayar biaya kuliah tepat waktu memiliki kecenderungan sangat tinggi untuk lulus (2.180 lulus). Sebaliknya, mahasiswa yang memiliki tunggakan (Debtor) memiliki proporsi dropout yang lebih tinggi (312 dropout) dibandingkan yang tidak memiliki tunggakan.
- Penerima Beasiswa: Jumlah mahasiswa dropout pada kelompok non-penerima beasiswa (1.287) jauh lebih tinggi dibandingkan pada kelompok penerima beasiswa (134), mengindikasikan bahwa bantuan finansial berperan penting dalam kelancaran studi.
- Latar Belakang Orang Tua: Mayoritas ibu dari mahasiswa memiliki pendidikan terakhir setingkat SMA (Secondary education) dan pekerjaan sebagai Unskilled Workers. Hal ini menunjukkan adanya potensi tantangan dari latar belakang sosio-ekonomi.
- Program Studi: Program studi Keperawatan (Nursing), Manajemen (Management), dan Layanan Sosial (Social Service) merupakan program dengan jumlah mahasiswa terbanyak.

## Menjalankan Sistem Machine Learning

Sebuah prototipe sistem machine learning telah dikembangkan dan dideploy sebagai aplikasi web interaktif menggunakan Streamlit. Sistem ini dapat diakses secara publik dan memungkinkan prediksi performa mahasiswa secara real-time.

Sistem ini menggunakan model XGBoost untuk memprediksi apakah seorang mahasiswa cenderung akan "Lulus" atau "Dropout" berdasarkan data yang diinput melalui formulir. Pengguna (misalnya, staf akademik) dapat mengisi informasi mahasiswa, dan aplikasi akan menampilkan hasil prediksi beserta probabilitasnya

Prototipe dapat diakses langsung melalui link berikut tanpa perlu instalasi:
https://syaokifaradisa9-student-performance-dicoding-app-ujnvb4.streamlit.app/

Jika ingin menjalankan aplikasi di komputer lokal, Anda dapat mengikuti langkah-langkah berikut dari repositori GitHub:

1. Clone repository dari Github:

    ````
      git clone https://github.com/syaokifaradisa9/Student-Performance-Dicoding.git
    ````

2. Masuk ke direktori proyek dan buat virtual environment baru:

    ````
      cd Student-Performance-Dicoding
      python -m venv env
      source env/bin/activate  # Pada Windows gunakan: env\Scripts\activate
    ````

3. Instal semua library yang dibutuhkan dari file requirements.txt:

    ````
      pip install -r requirements.txt
    ````

4. Jalankan aplikasi Streamlit:

    ````
      streamlit run app.py
    ````

## Conclusion

Jelaskan konklusi dari proyek yang dikerjakan.

### Rekomendasi Action Items

Berdasarkan temuan dari analisa data yang telah dilakukan ini, berikut adalah beberapa rekomendasi tindakan yang dapat diambil oleh Jaya Jaya Institut:

- Implementasi Program Bantuan Finansial Proaktif: Menggunakan data status pembayaran UKT dan status hutang untuk secara proaktif mengidentifikasi mahasiswa yang kesulitan secara finansial.  Tawarkan mereka skema pembayaran yang lebih fleksibel, konseling keuangan, atau kemudahan akses informasi beasiswa.
- Sistem Peringatan Dini dan Pendampingan Akademik: Mengintegrasikan sistem prediksi untuk menandai mahasiswa yang memiliki probabilitas dropout tinggi di akhir semester pertama dan kedua.  Mahasiswa yang ditandai harus secara otomatis dimasukkan ke dalam program pendampingan intensif bersama dosen wali atau mentor mahasiswa senior.

- Penguatan Program Dukungan Mahasiswa Generasi Pertama: Mengingat banyak orang tua mahasiswa yang tidak mengenyam pendidikan tinggi, institusi disarankan untuk membuat program dukungan yang dirancang khusus untuk mahasiswa yang merupakan orang pertama di keluarganya yang berkuliah (first-generation students).
