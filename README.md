📰 Fake News Detection Dashboard

UAP Machine Learning

📌 Deskripsi Proyek

Project ini merupakan aplikasi Fake News Detection berbasis Machine Learning & Deep Learning yang dibuat untuk memenuhi tugas Ujian Akhir Praktikum (UAP).
Aplikasi dibangun menggunakan Streamlit dan mendukung beberapa model klasifikasi teks untuk mendeteksi apakah sebuah berita tergolong FAKE atau REAL.

Model yang digunakan:

LSTM (Deep Learning)

BERT

DistilBERT

Ensemble (Majority Voting)

🎯 Tujuan

Mengimplementasikan model Machine Learning & Deep Learning untuk klasifikasi teks

Membandingkan performa beberapa model NLP

Menyediakan dashboard interaktif berbasis web menggunakan Streamlit

🧠 Model & Akurasi
Model	Akurasi
LSTM	81%
BERT	96%
DistilBERT	96%
Ensemble	Lebih stabil (Majority Vote)

🏆 Best Model: BERT

⚙️ Fitur Aplikasi

🔍 Prediksi berita satu teks

📂 Prediksi batch dari file CSV

📊 Menampilkan confidence score

🔁 Ensemble prediction (majority voting)

🖥️ Tampilan dashboard interaktif

🛠️ Teknologi yang Digunakan

Python

Streamlit

TensorFlow / Keras

PyTorch

Transformers (HuggingFace)

NumPy

Pandas

Scikit-learn

Matplotlib

Pillow

UAPMLC/
│
├── project/
│   ├── app.py
│   ├── models/
│   │   ├── lstm/
│   │   ├── bert/          # tidak disertakan (large files)
│   │   └── distilbert/    # tidak disertakan (large files)
│   ├── assets/
│   │   └── as.jpg
│
├── requirements.txt
├── README.md
└── .gitignore

📦 Instalasi

Clone repository:

git clone https://github.com/Masoby/UAPMLC.git
cd UAPMLC


(Opsional) Buat virtual environment:

python -m venv venv
venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

▶️ Menjalankan Aplikasi
streamlit run project/app.py


Aplikasi akan berjalan di browser:

http://localhost:8501

👤 Author

Nama: Mahrus Mahbubi
NIM: 202210370311411
Mata Kuliah: Machine Learning
Tugas: UAP
