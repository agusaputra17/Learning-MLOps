# Submission 1: IMDB Reviews Sentiment Analysis Pipeline Using TFX and Docker
Nama: Agus Saputra Kambea

Username dicoding: agus_saputra_kambea

| | Deskripsi |
| ----------- | ----------- |
| **Dataset** | TensorFlow Datasets - IMDB Reviews |
| **Masalah** | Dalam dunia hiburan, ulasan film dapat mempengaruhi keputusan penonton sebelum menonton film. Menganalisis ribuan ulasan secara manual tidak praktis dan memakan banyak waktu. Oleh karena itu, proyek ini bertujuan untuk membangun model machine learning yang dapat mengotomatiskan klasifikasi sentimen dari ulasan film berdasarkan teks yang diberikan. |
| **Solusi machine learning** | Solusi yang dibuat adalah model klasifikasi teks berbasis deep learning menggunakan Neural Network. Model ini akan memproses teks ulasan, mengubahnya menjadi representasi numerik, dan memprediksi apakah ulasan tersebut positif atau negatif. Target dari solusi ini adalah mencapai akurasi tinggi dalam mengklasifikasikan sentimen ulasan film. |
| **Metode pengolahan** | **Data Ingestion**: Mengambil dataset dari TensorFlow Dataset (tfds). **Pembagian Data**: Dataset dibagi menjadi training (90%), validation (10%). **Preprocessing**: Tokenisasi teks menggunakan TextVectorization, normalisasi teks (menghapus tanda baca dan mengubah huruf menjadi kecil), serta padding sequences untuk menyamakan panjang input. |
| **Tuning Hyperparameter** | Pada tahap ini digunakan komponen **Tuner** dari TFX untuk melakukan pencarian hyperparameter terbaik secara otomatis menggunakan KerasTuner (RandomSearch). Hyperparameter yang di-tuning antara lain: dimensi embedding, jumlah unit pada dense layer, dan learning rate. Proses tuning dilakukan pada data yang telah dipreprocessing, dan hasil hyperparameter terbaik digunakan pada tahap pelatihan model. |
| **Arsitektur model** | Model yang digunakan berbasis **Deep Neural Network (DNN)** dengan arsitektur sebagai berikut: **Embedding Layer**: Mengubah teks menjadi vektor numerik berdimensi 16. **GlobalAveragePooling1D**: Mengurangi dimensi output dari `embedding layer`. **Dense Layer 1** : 64 neuron dengan aktivasi `ReLU`. **Dense Layer 2** : 32 neuron dengan aktivasi `ReLU`. **Output Layer**: 1 neuron dengan aktivasi sigmoid untuk klasifikasi biner. **Optimasi**: Menggunakan `Adam Optimizer (learning rate = 0.01)` dengan fungsi `loss binary cross-entropy`. |
| **Metrik evaluasi** | Model dievaluasi menggunakan `Binary Accuracy` untuk mengukur seberapa baik model dalam mengklasifikasikan sentimen ulasan. Selain itu, `loss function (binary cross-entropy)` digunakan untuk mengukur kesalahan model. |
| **Performa model** | Model mencapai binary accuracy sebesar 97.78% dengan loss sebesar 0.0570 pada dataset pengujian. Hasil ini menunjukkan bahwa model memiliki performa yang sangat baik dalam mengklasifikasikan sentimen ulasan film. |

---
## Setup Environment

1. **Buat environment baru (opsional, direkomendasikan):**
    - **Menggunakan conda:**
      ```bash
      conda create --name mlops-tfx python==3.9.15
      conda activate mlops-tfx
      ```
    - **Menggunakan venv (alternatif):**
      ```bash
      python -m venv venv
      ```
      Aktifkan environment:
      - **Windows:**  
        ```
        venv\Scripts\activate
        ```
      - **Linux/Mac:**  
        ```
        source venv/bin/activate
        ```
2. **Aktifkan environment:**
    - **Jika menggunakan conda:**
      ```bash
      conda activate mlops-tfx
      ```
    - **Jika menggunakan venv:**
      - **Windows:**  
        ```
        venv\Scripts\activate
        ```
      - **Linux/Mac:**  
        ```
        source venv/bin/activate
        ```
3. **Install library dari `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Tahapan Pipeline

1. **Data Ingestion**  
   Mengambil dan memproses dataset IMDB Reviews dari TensorFlow Datasets.

2. **Data Validation**  
   Mengecek kualitas data, menghasilkan statistik, skema, dan mendeteksi anomali.

3. **Data Preprocessing**  
   Melakukan transformasi data (contoh: lowercasing, konversi tipe data) menggunakan modul `imdb_reviews_transform.py`.

4. **Tuner (Hyperparameter Tuning)**  
   Melakukan pencarian hyperparameter terbaik secara otomatis menggunakan komponen `Tuner` dari TFX dan KerasTuner (RandomSearch).  
   Hyperparameter yang di-tuning meliputi: dimensi embedding, jumlah unit dense layer, dan learning rate.

5. **Trainer (Model Training)**  
   Melatih model menggunakan data yang telah dipreprocessing dan hyperparameter terbaik dari tahap tuning.

6. **Evaluator**  
   Mengevaluasi performa model menggunakan metrik seperti binary accuracy dan AUC.

7. **Pusher (Deployment)**  
   Mendepoy model terbaik ke direktori serving agar dapat digunakan untuk inferensi.

---

## Troubleshooting

Jika mendapatkan error berikut saat menjalankan `imdb_reviews_transform.py`:
```
RuntimeError: Failed to build wheel
```
Silahkan install library berikut:
```
pip install wheel==0.43.0 setuptools==69.5.1
```

## Docker Usage

**Build Docker image:**
```
docker build -t imdb-reviews-model .
```

**Jalankan Docker container:**
```
docker run -p 8080:8501 imdb-reviews