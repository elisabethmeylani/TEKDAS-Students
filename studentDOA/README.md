# StudentDOA: Data + ML + LLM

Proyek ini mengikuti alur aplikasi contoh di folder `simulation`, disesuaikan untuk dataset mahasiswa dengan tiga kelas target: `Dropout`, `Graduate`, dan `Enrolled`.

## Isi folder

- `studentDOA.csv`: dataset sumber (4.424 baris, 35 kolom).
- `data_prep.py`: validasi data, pemisahan train/test terstratifikasi, dan laporan kualitas data.
- `ml_model.py`: dua pipeline Random Forest multiclass (data hingga semester 1 dan hingga semester 2), evaluasi, dan keluaran prediksi/feature importance.
- `vai_analyst.py`: prompt dan analisis opsional dengan Ollama.
- `app.py`: dashboard Streamlit untuk melihat data, ringkasan akademik, prediksi, dan interpretasi LLM.

## Menjalankan

Aktifkan virtual environment, lalu dari folder `studentDOA`:

```powershell
pip install -r requirements.txt
python data_prep.py
python ml_model.py
streamlit run app.py
```

Untuk memakai tab LLM, jalankan Ollama dan unduh modelnya pada terminal terpisah:

```powershell
ollama serve
ollama pull llama3.2
```

Dashboard data, BI, dan ML dapat digunakan tanpa Ollama. Kolom kategori dalam dataset banyak yang berbentuk kode angka; jangan mengartikannya sebagai nilai berurutan.

Di sidebar aplikasi, pilih catatan mahasiswa dan batas data yang tersedia. Pilihan Semester 1 atau Semester 2 mengubah fitur yang ditampilkan dan model yang dipakai. Jalankan `ml_model.py` untuk membuat kedua model.

## Batas interpretasi model

Model Semester 1 hanya menggunakan informasi pendaftaran, latar belakang, ekonomi, dan akademik semester 1. Model Semester 2 menggunakan informasi hingga semester 2. Prediksi tetap merupakan perkiraan, bukan bukti sebab-akibat.
