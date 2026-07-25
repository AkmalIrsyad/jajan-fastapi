# -*- coding: utf-8 -*-
"""
===============================================================
 TRAINING MODEL TF-IDF + KNN + COSINE SIMILARITY
 Sistem Rekomendasi Kuliner Kota Bekasi — JajanBekasi!
===============================================================
 Alur:
   1. Load & preprocessing dataset GoFood + kuesioner
   2. Bangun dokumen TF-IDF per item kuliner
   3. Fit TF-IDF Vectorizer → sparse matrix
   4. Fit KNN (cosine, brute) di atas TF-IDF matrix
   5. Evaluasi Precision@5 & Coverage
   6. Simpan model ke model_tfidf.pkl
===============================================================
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors          # ✅ KNN


# =========================================================
# PATH
# =========================================================
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH   = os.path.join(BASE_DIR, "data", "gofood_bekasi_kota.csv")
KUESIONER_PATH = os.path.join(BASE_DIR, "data", "ResponsKeywordAnalysis.csv")
MODEL_OUT      = os.path.join(BASE_DIR, "model_tfidf.pkl")
REPORT_OUT     = os.path.join(BASE_DIR, "evaluasi_model.json")


# =========================================================
# PRICE MAP
# =========================================================
PRICE_MAP = {
    1: "10rb - 25rb",
    2: "25rb - 50rb",
    3: "50rb - 100rb",
    4: ">100rb"
}


# =========================================================
# KECAMATAN MAPPING (kelurahan → kecamatan)
# =========================================================
KECAMATAN_MAPPING = {
    # Bekasi Barat
    "bintara":          "Bekasi Barat",
    "bintarajaya":      "Bekasi Barat",
    "bintara jaya":     "Bekasi Barat",
    "jakasampurna":     "Bekasi Barat",
    "kranji":           "Bekasi Barat",
    "kotabaru":         "Bekasi Barat",
    "kota baru":        "Bekasi Barat",

    # Bekasi Timur
    "aren jaya":        "Bekasi Timur",
    "arenjaya":         "Bekasi Timur",
    "bekasi jaya":      "Bekasi Timur",
    "bekasijaya":       "Bekasi Timur",
    "duren jaya":       "Bekasi Timur",
    "durenjaya":        "Bekasi Timur",
    "margahayu":        "Bekasi Timur",
    "rawasemut":        "Bekasi Timur",
    "bulak kapal":      "Bekasi Timur",
    "bulakkapal":       "Bekasi Timur",

    # Bekasi Selatan
    "jakamulya":        "Bekasi Selatan",
    "jakasetia":        "Bekasi Selatan",
    "kayuringinjaya":   "Bekasi Selatan",
    "kayuringin jaya":  "Bekasi Selatan",
    "margajaya":        "Bekasi Selatan",
    "marga jaya":       "Bekasi Selatan",
    "pekayon":          "Bekasi Selatan",
    "pekayon jaya":     "Bekasi Selatan",
    "pekayonjaya":      "Bekasi Selatan",

    # Bekasi Utara
    "harapan baru":     "Bekasi Utara",
    "harapanbaru":      "Bekasi Utara",
    "harapan jaya":     "Bekasi Utara",
    "harapanjaya":      "Bekasi Utara",
    "kaliabang tengah": "Bekasi Utara",
    "kaliabangtengah":  "Bekasi Utara",
    "marga mulya":      "Bekasi Utara",
    "margamulya":       "Bekasi Utara",
    "perwira":          "Bekasi Utara",
    "teluk pucung":     "Bekasi Utara",
    "telukpucung":      "Bekasi Utara",
    "summarecon bekasi":"Bekasi Utara",
    "pondok ungu permai":"Bekasi Utara",

    # Rawalumbu
    "bojong rawalumbu": "Rawalumbu",
    "bojongrawalumbu":  "Rawalumbu",
    "bojong menteng":   "Rawalumbu",
    "bojongmenteng":    "Rawalumbu",
    "pengasinan":       "Rawalumbu",
    "sepanjangjaya":    "Rawalumbu",
    "sepanjang jaya":   "Rawalumbu",
    "narogong":         "Rawalumbu",
    "pondok hijau":     "Rawalumbu",

    # Mustika Jaya
    "cimuning":         "Mustika Jaya",
    "mustikajaya":      "Mustika Jaya",
    "mustika jaya":     "Mustika Jaya",
    "mustikasari":      "Mustika Jaya",
    "pedurenan":        "Mustika Jaya",
    "padurenan":        "Mustika Jaya",
    "mutiara gading":   "Mustika Jaya",

    # Bantargebang
    "bantargebang":     "Bantargebang",
    "ciketing udik":    "Bantargebang",
    "ciketingudik":     "Bantargebang",
    "cikiwul":          "Bantargebang",
    "sumur batu":       "Bantargebang",
    "sumurbatu":        "Bantargebang",

    # Pondok Melati
    "pondok melati":    "Pondok Melati",
    "pondokmelati":     "Pondok Melati",
    "jatimelati":       "Pondok Melati",
    "jatimurni":        "Pondok Melati",
    "jatirahayu":       "Pondok Melati",
    "jatiwarna":        "Pondok Melati",

    # Pondok Gede
    "pondok gede":      "Pondok Gede",
    "pondokgede":       "Pondok Gede",
    "jatibening":       "Pondok Gede",
    "jatibening baru":  "Pondok Gede",
    "jatibeningbaru":   "Pondok Gede",
    "jaticempaka":      "Pondok Gede",
    "jatimakmur":       "Pondok Gede",
    "jatiwaringin":     "Pondok Gede",

    # Jatisampurna
    "jatisampurna":     "Jatisampurna",
    "jatikarya":        "Jatisampurna",
    "jatiraden":        "Jatisampurna",
    "jatirangga":       "Jatisampurna",
    "jatiranggon":      "Jatisampurna",
    "harjamukti":       "Jatisampurna",

    # Jatiasih
    "jatiasih":         "Jatiasih",
    "jatikramat":       "Jatiasih",
    "jatiluhur":        "Jatiasih",
    "jatimekar":        "Jatiasih",
    "jatirasa":         "Jatiasih",
    "jatisari":         "Jatiasih",

    # Medan Satria
    "harapan mulya":    "Medan Satria",
    "harapanmulya":     "Medan Satria",
    "kalibaru":         "Medan Satria",
    "kali baru":        "Medan Satria",
    "medansatria":      "Medan Satria",
    "medan satria":     "Medan Satria",
    "pejuang":          "Medan Satria",
    "kota harapan indah": "Medan Satria",
}


# =========================================================
# CONFIG TF-IDF
# =========================================================
NGRAM_RANGE  = (1, 2)
MAX_FEATURES = 1500

BOBOT = {
    "nama":        2,
    "kategori":    3,
    "kecamatan":   2,
    "price_label": 2,
    "tags":        2,
    "deskripsi":   1,
}

# =========================================================
# CONFIG KNN
# =========================================================
KNN_N_NEIGHBORS = 20      # jumlah tetangga yang dicari saat retrieval
KNN_METRIC      = "cosine"
KNN_ALGORITHM   = "brute" # wajib 'brute' untuk sparse matrix TF-IDF


# =========================================================
# STOPWORDS
# =========================================================
STOPWORDS = {
    "dan","di","ke","yang","dari","untuk","dengan","adalah","ini",
    "itu","ada","atau","juga","sudah","akan","pada","dalam","tidak",
    "kami","kita","bisa","saya","anda","nya","bekasi","kota",
    "restoran","warung","kedai","makan","makanan","tempat","menu",
}


# =========================================================
# LOAD DATA
# =========================================================
def load_data():
    print("\n=== [1/5] LOAD DATA ===")

    df = pd.read_csv(DATASET_PATH, low_memory=False)

    # ── PRICE LABEL ──
    if "price_label" not in df.columns:
        if "price_level" in df.columns:
            df["price_label"] = df["price_level"].map(PRICE_MAP)
        else:
            df["price_label"] = "25rb - 50rb"

    df["price_label"] = df["price_label"].fillna("25rb - 50rb")

    # ── KOLOM WAJIB ──
    for col in ["tags", "deskripsi", "kategori", "kecamatan", "nama", "address"]:
        if col not in df.columns:
            df[col] = ""

    df["is_kuesioner"] = 0

    # ── LOAD KUESIONER ──
    if os.path.exists(KUESIONER_PATH):
        print("   → Load kuesioner...")
        try:
            df_q = pd.read_csv(KUESIONER_PATH)
        except Exception:
            df_q = pd.DataFrame()

        if not df_q.empty:
            kolom_nama = None
            for k in ["nama", "nama_kuliner", "kuliner", "rekomendasi"]:
                if k in df_q.columns:
                    kolom_nama = k
                    break

            if kolom_nama:
                df_q = df_q.rename(columns={kolom_nama: "nama"})
                df_q["nama"]        = df_q["nama"].astype(str).str.strip()
                df_q                = df_q[df_q["nama"].str.len() > 2]
                df_q["kategori"]    = df_q.get("kategori", "Kuesioner")
                df_q["kecamatan"]   = df_q.get("kecamatan", "")
                df_q["price_label"] = df_q.get("price_label", "25rb - 50rb")
                df_q["price_label"] = df_q["price_label"].astype(str).str.replace("k", "rb", regex=False)
                df_q["tags"]        = ""
                df_q["deskripsi"]   = df_q.get("deskripsi", "")
                df_q["is_kuesioner"] = 1

                df_q = df_q.reindex(columns=df.columns, fill_value="")
                df   = pd.concat([df, df_q], ignore_index=True)

    # ── NORMALISASI KECAMATAN ──
    df["kecamatan"] = (
        df["kecamatan"].astype(str).str.strip().str.lower()
        .map(KECAMATAN_MAPPING)
        .fillna(df["kecamatan"].astype(str).str.strip().str.lower())
    )

    df = df.fillna("")
    print(f"   Total data: {len(df)}")
    return df


# =========================================================
# CLEAN TEXT
# =========================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(t for t in text.split() if t not in STOPWORDS)


# =========================================================
# BUILD DOCUMENT (weighted)
# =========================================================
def build_doc(row):
    parts = []
    for col, w in BOBOT.items():
        txt = clean_text(row.get(col, ""))
        if txt:
            parts.extend([txt] * w)
    return " ".join(parts)


# =========================================================
# PREPROCESS
# =========================================================
def preprocess(df):
    print("\n=== [2/5] PREPROCESS ===")
    df["doc"] = df.apply(build_doc, axis=1)
    df = df[df["doc"].str.len() > 5].reset_index(drop=True)
    print(f"   Data setelah cleaning: {len(df)}")
    return df


# =========================================================
# TRAIN TF-IDF + KNN
# =========================================================
def train(df):
    # ── TF-IDF ──
    print("\n=== [3/5] TRAIN TF-IDF ===")
    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(df["doc"])
    print(f"   TF-IDF shape: {tfidf_matrix.shape}")

    # ── KNN ──
    print("\n=== [4/5] TRAIN KNN ===")
    knn_model = NearestNeighbors(
        n_neighbors=KNN_N_NEIGHBORS,
        metric=KNN_METRIC,
        algorithm=KNN_ALGORITHM,   # 'brute' wajib untuk sparse matrix
        n_jobs=-1
    )
    knn_model.fit(tfidf_matrix)
    print(f"   KNN fitted: n_neighbors={KNN_N_NEIGHBORS}, metric={KNN_METRIC} ✅")

    return vectorizer, tfidf_matrix, knn_model


# =========================================================
# EVALUASI
# =========================================================
def evaluasi(df, tfidf_matrix, top_k=5):
    print("\n=== [5/5] EVALUASI ===")

    N          = len(df)
    idx_sample = np.random.choice(N, min(50, N), replace=False)

    precision_list = []
    coverage_set   = set()

    for idx in idx_sample:
        sim     = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        sim[idx] = -1
        top_idx  = np.argsort(sim)[::-1][:top_k]

        kategori = str(df.iloc[idx]["kategori"]).lower()
        relevan  = sum(
            1 for i in top_idx
            if kategori in str(df.iloc[i]["kategori"]).lower()
        )

        precision_list.append(relevan / top_k)
        coverage_set.update(top_idx)

    result = {
        "precision@5": round(float(np.mean(precision_list)), 4),
        "coverage":    round(len(coverage_set) / N, 4),
        "total_data":  N,
        "knn_neighbors": KNN_N_NEIGHBORS,
        "knn_metric":    KNN_METRIC,
    }
    print(f"   {result}")
    return result


# =========================================================
# SAVE MODEL
# =========================================================
def save_model(vectorizer, tfidf_matrix, df, eval_result, knn_model):
    joblib.dump({
        "vectorizer":   vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "df":           df,
        "knn_model":    knn_model,        # ✅ KNN tersimpan
    }, MODEL_OUT)

    with open(REPORT_OUT, "w") as f:
        json.dump(eval_result, f, indent=2)

    print(f"\n✅ Model saved → {MODEL_OUT}")
    print(f"✅ Evaluasi  → {REPORT_OUT}")


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 55)
    print(" TRAINING JAJANBEKASI! — TF-IDF + KNN")
    print("=" * 55)

    df                           = load_data()
    df                           = preprocess(df)
    vectorizer, tfidf_matrix, knn_model = train(df)
    eval_result                  = evaluasi(df, tfidf_matrix)
    save_model(vectorizer, tfidf_matrix, df, eval_result, knn_model)

    print("\n🎉 DONE — model siap dipakai oleh app.py")


if __name__ == "__main__":
    main()