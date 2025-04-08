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

## Data Preparation
Pada tahap ini, dilakukan sejumlah proses persiapan data agar dapat digunakan secara optimal oleh algoritma machine learning. Pertama, fitur kategori dikonversi menjadi numerik menggunakan teknik encoding seperti LabelEncoder, agar dapat diproses oleh model. Selanjutnya, dilakukan standarisasi menggunakan StandardScaler untuk menyamakan skala antar fitur, yang penting untuk model berbasis jarak atau regresi.

Kemudian, dilakukan reduksi dimensi dengan Principal Component Analysis (PCA) untuk menyederhanakan kompleksitas data tanpa kehilangan informasi penting. Terakhir, dataset dibagi menggunakan fungsi train_test_split dengan rasio 80:20 menjadi data latih dan data uji, guna melatih dan mengevaluasi model secara adil.

Langkah-langkah ini diperlukan untuk meningkatkan performa model, mencegah overfitting, serta memastikan data dalam kondisi bersih dan terstruktur.

## Modeling

Pada tahap ini, dilakukan pembangunan model machine learning untuk menyelesaikan permasalahan klasifikasi diagnosis PCOS. Empat algoritma digunakan dalam proses ini, yaitu **Logistic Regression**, **Decision Tree**, **Random Forest**, dan **XGBoost**. Pemilihan keempat algoritma ini didasarkan pada pertimbangan performa, interpretabilitas, serta kemampuan dalam menangani dataset dengan fitur numerik dan kategorikal.

**Logistic Regression** digunakan sebagai baseline karena bersifat sederhana dan mudah diinterpretasikan. Model ini cocok digunakan untuk permasalahan klasifikasi biner, namun memiliki keterbatasan dalam menangani data yang kompleks dan bersifat non-linear.

**Decision Tree** dipilih karena kemampuannya dalam menangani kombinasi fitur numerik dan kategorikal serta kemudahan interpretasi model. Namun, model ini cenderung overfitting jika tidak dilakukan pemangkasan atau pengaturan parameter yang tepat.

**Random Forest**, sebagai model ensemble berbasis banyak pohon keputusan, memiliki keunggulan dalam meningkatkan akurasi dan mengurangi risiko overfitting. Untuk meningkatkan performanya, dilakukan proses tuning terhadap beberapa parameter seperti jumlah estimator, kedalaman maksimum pohon, dan jumlah sampel minimum untuk pemisahan dan daun.

**XGBoost** dipilih karena dikenal memiliki performa yang sangat baik pada berbagai permasalahan klasifikasi. Selain itu, algoritma ini juga dilengkapi dengan fitur regularisasi untuk mencegah overfitting. Setelah membangun model awal, dilakukan hyperparameter tuning untuk meningkatkan hasil prediksi, termasuk penyesuaian terhadap jumlah estimasi, laju pembelajaran, kedalaman maksimum pohon, dan rasio subsampling.

Setelah dilakukan pelatihan dan evaluasi terhadap seluruh model, diperoleh hasil bahwa **XGBoost tanpa tuning memberikan performa terbaik** dengan *accuracy* sebesar **96%**, *precision* **91,89%**, *recall* **87,18%**, dan *F1-score* **89,47%**. Meskipun model XGBoost yang telah dituning masih menunjukkan performa tinggi, terjadi sedikit penurunan pada accuracy dan F1-score dibandingkan model awal, sehingga tuning tidak selalu menjamin peningkatan performa secara signifikan.

Dengan mempertimbangkan keseimbangan antara akurasi dan kemampuan dalam mendeteksi kasus positif yang tinggi—yang sangat penting dalam konteks diagnosis medis—maka **model XGBoost tanpa tuning dipilih sebagai model terbaik** dalam proyek ini.


## Evaluation

Untuk mengevaluasi performa model dalam proyek klasifikasi diagnosis PCOS ini, digunakan beberapa metrik evaluasi yang umum digunakan dalam permasalahan klasifikasi biner, yaitu **accuracy**, **precision**, **recall**, dan **F1 score**. Pemilihan metrik ini mempertimbangkan konteks medis dari proyek, di mana kesalahan klasifikasi terutama terhadap kasus positif (yaitu pasien dengan risiko PCOS) dapat berdampak serius.

### Penjelasan Metrik Evaluasi

- **Accuracy** mengukur proporsi prediksi yang benar dibandingkan dengan total keseluruhan data. Rumus:

  $$
  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
  $$

- **Precision** adalah rasio antara jumlah prediksi positif yang benar (*True Positive*) terhadap seluruh prediksi positif. Precision penting ketika *false positive* harus diminimalkan, misalnya untuk menghindari overdiagnosis.

  $$
  \text{Precision} = \frac{TP}{TP + FP}
  $$

- **Recall** (atau *Sensitivity*) mengukur seberapa baik model dalam menemukan semua kasus positif. Ini penting dalam konteks medis, di mana kesalahan tidak mendeteksi kasus positif (*False Negative*) bisa berbahaya.

  $$
  \text{Recall} = \frac{TP}{TP + FN}
  $$

- **F1 Score** adalah rata-rata harmonik dari precision dan recall. F1 score memberikan gambaran menyeluruh terhadap keseimbangan kedua metrik tersebut.

  $$
  \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  $$


### Hasil Evaluasi

Model terbaik dari proyek ini adalah **XGBoost tanpa tuning**, dengan hasil evaluasi sebagai berikut:

- **Accuracy**: 96%
- **Precision**: 91,89%
- **Recall**: 87,18%
- **F1 Score**: 89,47%

Hasil ini menunjukkan bahwa model mampu mengklasifikasikan sebagian besar kasus dengan benar, terutama dalam mendeteksi pasien yang berisiko PCOS (recall tinggi). Precision yang tinggi juga menunjukkan bahwa sebagian besar prediksi positif memang benar adanya. F1 score yang tinggi mengindikasikan bahwa model memiliki keseimbangan yang baik antara precision dan recall, menjadikannya pilihan ideal untuk digunakan dalam konteks diagnosis awal PCOS.

Dengan demikian, metrik evaluasi mendukung bahwa model XGBoost dapat diandalkan dalam membantu deteksi dini PCOS dengan akurasi tinggi dan tingkat kesalahan yang rendah.

