# Laporan Proyek Machine Learning - Tarisa Nur Safitri

## Domain Proyek

Polycystic Ovary Syndrome (PCOS) merupakan gangguan hormon yang kompleks dan umum terjadi, memengaruhi sekitar 6–10% perempuan usia reproduktif di dunia. Kondisi ini dapat menyebabkan infertilitas, resistensi insulin, hingga risiko penyakit metabolik. Namun, diagnosis PCOS sering terlambat karena gejalanya yang bervariasi dan menyerupai gangguan lain. Keterlambatan ini berdampak pada penanganan yang tidak optimal dan meningkatnya risiko komplikasi di masa depan.

Penggunaan machine learning dapat menjadi solusi untuk mengatasi tantangan tersebut dengan memanfaatkan data klinis pasien untuk mengidentifikasi pola dan risiko secara otomatis. Pada proyek ini, digunakan dataset berisi 1000 data pasien dengan lima fitur utama yang berhubungan dengan PCOS untuk membangun model klasifikasi diagnosis. Model ini diharapkan dapat membantu mempercepat proses skrining dan meningkatkan akurasi prediksi, sehingga mendukung pengambilan keputusan medis secara lebih efisien.

Referensi:
- [Recommendations from the international evidence-based guideline for the assessment and management of polycystic ovary syndrome. Human Reproduction, 33(9), 1602–1618.](https://academic.oup.com/humrep/article/33/9/1602/5056069)
- [Polycystic ovary syndrome. Nature Reviews Disease Primers, 2, 16057.](https://www.nature.com/articles/nrdp201657)


## Business Understanding

### Problem Statements

1. Diagnosis PCOS saat ini masih sering bergantung pada kombinasi gejala subjektif dan pemeriksaan lanjutan, yang membutuhkan waktu serta biaya, sehingga menyulitkan deteksi dini, terutama di fasilitas kesehatan dengan keterbatasan sumber daya.

2. Dalam banyak kasus, tidak semua data klinis berperan signifikan dalam diagnosis. Mengetahui fitur yang paling berpengaruh dapat membantu tenaga medis fokus pada indikator yang benar-benar relevan.

3. Salah satu tantangan dalam adopsi machine learning di dunia medis adalah keterbatasan interpretabilitas model, sehingga penting untuk mengembangkan model yang dapat dipahami dan dipercaya oleh praktisi kesehatan.

### Goals

1. Mengembangkan model klasifikasi berbasis machine learning yang mampu memprediksi diagnosis PCOS secara akurat hanya dengan menggunakan lima fitur utama.
Hal ini bertujuan untuk mempercepat proses skrining awal dan membantu pengambilan keputusan medis tanpa perlu pemeriksaan lanjutan yang mahal.

2. Mengidentifikasi fitur-fitur paling penting yang berkontribusi dalam prediksi diagnosis PCOS.
Dengan melakukan feature importance analysis, proyek ini akan memberikan insight yang lebih mendalam tentang variabel-variabel yang paling relevan dalam menentukan diagnosis.

3. Membangun model klasifikasi yang tidak hanya akurat, namun juga dapat dijelaskan (interpretable), seperti decision tree atau model dengan SHAP values.
Model yang interpretable akan memudahkan dokter atau ahli kesehatan untuk memahami alasan di balik setiap prediksi, sehingga meningkatkan kepercayaan dalam penggunaannya.

### Solution statements
1. Menguji dan membandingkan beberapa algoritma machine learning seperti Logistic Regression, Decision Tree, Random Forest, dan XGBoost untuk membangun model klasifikasi diagnosis PCOS. Perbandingan dilakukan berdasarkan metrik evaluasi seperti akurasi, precision, recall, dan F1-score.

2. Melakukan hyperparameter tuning pada model terbaik menggunakan Grid Search atau Random Search untuk meningkatkan performa dan generalisasi model.

3. Menggunakan metode explainable AI (XAI) seperti SHAP (SHapley Additive exPlanations) atau feature importance dari tree-based models untuk menginterpretasikan hasil model dan memahami kontribusi setiap fitur terhadap prediksi.

4. Membagi data menjadi train-test split yang representatif dan menggunakan teknik validasi silang (cross-validation) untuk memastikan performa model stabil dan tidak overfitting.


## Data Understanding
Dataset yang digunakan dalam proyek ini adalah dataset berjudul "PCOS Dataset" yang berisi informasi klinis terkait diagnosis Polycystic Ovary Syndrome (PCOS) pada perempuan usia reproduktif. Dataset ini terdiri dari 1000 entri data pasien, dengan masing-masing entri mewakili satu pasien, serta memuat 5 fitur utama yang umum dijadikan indikator risiko PCOS. Dataset ini digunakan untuk membangun model klasifikasi diagnosis PCOS berbasis machine learning. 
Sumber : [PCOS Diagnosis Dataset - Kaggle](https://www.kaggle.com/datasets/samikshadalvi/pcos-diagnosis-dataset).  

### Variabel-variabel pada PCOS Diagnosis Dataset adalah sebagai berikut:
- BMI: Indeks massa tubuh pasien, dihitung dari berat dan tinggi badan. Nilai yang tinggi sering dikaitkan dengan risiko PCOS yang lebih besar.
- Testosterone: Tingkat hormon testosteron dalam tubuh pasien. Kadar yang lebih tinggi dari normal merupakan salah satu indikator utama PCOS.
- Menstrual Irregularity: Menunjukkan keteraturan siklus menstruasi. Nilai yang menunjukkan ketidakteraturan dapat menjadi gejala PCOS.
- Antral Follicle Count (AFC): Jumlah folikel antral dalam ovarium. Jumlah folikel yang tinggi biasanya ditemukan pada pasien dengan PCOS.
- Age: Usia pasien, yang dapat memengaruhi risiko dan gejala PCOS.

Target Variable:
- PCOS Diagnosis: Label biner yang menunjukkan apakah pasien didiagnosis menderita PCOS atau tidak. (0 = Tidak, 1 = Ya)

### Exploratory Data Analysis
![image](https://github.com/user-attachments/assets/a8581ba7-e0a1-451e-bf9a-5675c20b75cd)
- Distribusi Diagnosis PCOS menunjukkan adanya ketidakseimbangan antara jumlah pasien yang terdiagnosis dan tidak terdiagnosis PCOS.
- Distribusi Usia memperlihatkan bahwa pasien dengan PCOS cenderung berada pada rentang usia muda hingga awal 30-an.
- Boxplot BMI, Testosteron, dan Jumlah Folikel Antral menunjukkan bahwa nilai-nilai ini cenderung lebih tinggi pada pasien yang didiagnosis PCOS.
- Heatmap Korelasi mengindikasikan adanya korelasi positif yang signifikan antara level testosteron dan jumlah folikel antral dengan diagnosis PCOS.
