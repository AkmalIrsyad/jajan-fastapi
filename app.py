from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os, joblib, re
from sklearn.metrics.pairwise import cosine_similarity

# ===============================================================
# CONFIG
# ===============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_tfidf.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "public")
FOTO_DIR   = os.path.join(BASE_DIR, "foto")

app = FastAPI(title="JajanBekasi! API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================================================
# GLOBAL
# ===============================================================
df           = None
tfidf_matrix = None
vectorizer   = None
knn_model    = None    # ✅ KNN model global

# ---------------------------------------------------------------
# BOBOT RANKING — dari hasil kuesioner (40 responden)
# Harga: 4.15/5 | Rating: 4.08/5 | Jarak: 3.70/5
# Total: 11.93 → dinormalisasi jadi bobot
# ---------------------------------------------------------------
BOBOT_HARGA  = 4.15 / 11.93   # ≈ 0.348
BOBOT_RATING = 4.08 / 11.93   # ≈ 0.342
BOBOT_CBF    = 0.310           # content similarity

PRICE_SCORE = {
    "10rb - 25rb":  1.0,
    "25rb - 50rb":  0.85,
    "50rb - 100rb": 0.50,
    ">100rb":       0.15,
}

KECAMATAN_MAPPING = {
    # Bantargebang
    "bantargebang":     "Bantargebang",
    "ciketing udik":    "Bantargebang",
    "ciketingudik":     "Bantargebang",
    "cikiwul":          "Bantargebang",
    "sumur batu":       "Bantargebang",
    "sumurbatu":        "Bantargebang",

    # Bekasi Barat
    "bekasi barat":     "Bekasi Barat",
    "bintara":          "Bekasi Barat",
    "bintara jaya":     "Bekasi Barat",
    "bintarajaya":      "Bekasi Barat",
    "jakasampurna":     "Bekasi Barat",
    "kota baru":        "Bekasi Barat",
    "kotabaru":         "Bekasi Barat",
    "kranji":           "Bekasi Barat",

    # Bekasi Selatan
    "bekasi selatan":   "Bekasi Selatan",
    "jakamulya":        "Bekasi Selatan",
    "jakasetia":        "Bekasi Selatan",
    "kayuringin jaya":  "Bekasi Selatan",
    "kayuringinjaya":   "Bekasi Selatan",
    "marga jaya":       "Bekasi Selatan",
    "margajaya":        "Bekasi Selatan",
    "pekayon jaya":     "Bekasi Selatan",
    "pekayonjaya":      "Bekasi Selatan",

    # Bekasi Timur
    "bekasi timur":     "Bekasi Timur",
    "aren jaya":        "Bekasi Timur",
    "arenjaya":         "Bekasi Timur",
    "bekasi jaya":      "Bekasi Timur",
    "bekasijaya":       "Bekasi Timur",
    "duren jaya":       "Bekasi Timur",
    "durenjaya":        "Bekasi Timur",
    "margahayu":        "Bekasi Timur",

    # Bekasi Utara
    "bekasi utara":     "Bekasi Utara",
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

    # Jatiasih
    "jatiasih":         "Jatiasih",
    "jatikramat":       "Jatiasih",
    "jatiluhur":        "Jatiasih",
    "jatimekar":        "Jatiasih",
    "jatirasa":         "Jatiasih",
    "jatisari":         "Jatiasih",

    # Jatisampurna
    "jatisampurna":     "Jatisampurna",
    "jatikarya":        "Jatisampurna",
    "jatiraden":        "Jatisampurna",
    "jatirangga":       "Jatisampurna",
    "jatiranggon":      "Jatisampurna",

    # Medan Satria
    "medan satria":     "Medan Satria",
    "medansatria":      "Medan Satria",
    "pejuang":          "Medan Satria",
    "harapan mulya":    "Medan Satria",
    "harapanmulya":     "Medan Satria",
    "kali baru":        "Medan Satria",
    "kalibaru":         "Medan Satria",

    # Mustika Jaya
    "mustika jaya":     "Mustika Jaya",
    "mustikajaya":      "Mustika Jaya",
    "cimuning":         "Mustika Jaya",
    "mustikasari":      "Mustika Jaya",
    "pedurenan":        "Mustika Jaya",

    # Pondok Gede
    "pondok gede":      "Pondok Gede",
    "pondokgede":       "Pondok Gede",
    "jatibening":       "Pondok Gede",
    "jatibening baru":  "Pondok Gede",
    "jatibeningbaru":   "Pondok Gede",
    "jaticempaka":      "Pondok Gede",
    "jatimakmur":       "Pondok Gede",
    "jatiwaringin":     "Pondok Gede",

    # Pondok Melati
    "pondok melati":    "Pondok Melati",
    "pondokmelati":     "Pondok Melati",
    "jatimelati":       "Pondok Melati",
    "jatimurni":        "Pondok Melati",
    "jatirahayu":       "Pondok Melati",
    "jatiwarna":        "Pondok Melati",

    # Rawalumbu
    "rawalumbu":        "Rawalumbu",
    "bojong menteng":   "Rawalumbu",
    "bojongmenteng":    "Rawalumbu",
    "bojong rawalumbu": "Rawalumbu",
    "bojongrawalumbu":  "Rawalumbu",
    "pengasinan":       "Rawalumbu",
    "sepanjang jaya":   "Rawalumbu",
    "sepanjangjaya":    "Rawalumbu",
}


# ===============================================================
# LOAD MODEL
# ===============================================================
def init_model():
    global df, tfidf_matrix, vectorizer, knn_model

    if not os.path.exists(MODEL_PATH):
        raise Exception("❌ Model belum ada! Jalankan model_training.py dulu.")

    payload      = joblib.load(MODEL_PATH)
    df           = payload["df"]
    tfidf_matrix = payload["tfidf_matrix"]
    vectorizer   = payload["vectorizer"]
    knn_model    = payload.get("knn_model")    # ✅ load KNN

    if knn_model is None:
        print("⚠️  knn_model tidak ditemukan di pkl — re-train diperlukan!")

    # ── NORMALISASI ──
    df["kecamatan"]   = df["kecamatan"].astype(str).str.strip().str.lower()
    df["kecamatan"]   = (
        df["kecamatan"].map(KECAMATAN_MAPPING)
        .fillna(df["kecamatan"])
        .str.title()
        .str.strip()
    )
    df["price_label"] = df["price_label"].astype(str).str.strip().str.lower()
    df["price_score"] = df["price_label"].map(PRICE_SCORE).fillna(0.3)

    max_r             = df["rating_avg"].max() if "rating_avg" in df.columns else 5.0
    df["rating_norm"] = (
        df["rating_avg"] / (max_r or 5.0)
        if "rating_avg" in df.columns else 0.5
    )

    # ── RELOAD FOTO DARI CSV ──
    foto_path = os.path.join(BASE_DIR, "data", "bekasi_kuliner_with_photos.csv")

    for col in ["photo_url", "photo_source", "photo_local", "nama_key"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if os.path.exists(foto_path):
        try:
            df_foto = pd.read_csv(
                foto_path,
                usecols=lambda c: c in ["displayName", "photo_url", "photo_source"]
            )
            df_foto["photo_url"] = df_foto["photo_url"].astype(str).str.strip()
            df_foto = df_foto[
                df_foto["photo_url"].notna() &
                (df_foto["photo_url"] != "") &
                (df_foto["photo_url"] != "nan") &
                df_foto["photo_url"].str.startswith("http")
            ]
            print(f"   CSV foto: {len(df_foto)} baris punya URL valid")

            df_foto = df_foto.rename(columns={"displayName": "nama"})
            df_foto["nama_key"] = df_foto["nama"].astype(str).str.lower().str.strip()
            df_foto = df_foto.drop_duplicates(subset="nama_key", keep="first")

            df["nama_key"] = df["nama"].astype(str).str.lower().str.strip()
            df = df.merge(
                df_foto[["nama_key", "photo_url", "photo_source"]],
                on="nama_key",
                how="left"
            )
            df.drop(columns=["nama_key"], inplace=True)

            matched = (
                df["photo_url"].notna() &
                (df["photo_url"] != "") &
                (df["photo_url"] != "nan")
            ).sum()
            print(f"✅ Foto dimuat dari CSV: {matched}/{len(df)} data punya foto")

        except Exception as e:
            df["photo_url"]    = None
            df["photo_source"] = None
            print(f"⚠️ Gagal load foto: {e}")
    else:
        df["photo_url"]    = None
        df["photo_source"] = None
        print("⚠️ File bekasi_kuliner_with_photos.csv tidak ditemukan")

    # ── KONVERSI photo_local → URL (fallback) ──
    if "photo_local" in df.columns:
        def local_to_url(val):
            if not val or pd.isna(val) or str(val).strip() in ("", "nan"):
                return None
            return f"https://api.jajanbekasi.web.id/foto/foto/{os.path.basename(str(val))}"

        mask = df["photo_url"].isna() | (df["photo_url"].astype(str).isin(["", "nan", "None"]))
        df.loc[mask, "photo_url"] = df.loc[mask, "photo_local"].apply(local_to_url)

    final_matched = df["photo_url"].notna().sum()
    print(f"✅ Model loaded: {len(df)} data | foto: {final_matched} | KNN: {'✅' if knn_model else '❌'}")


# ===============================================================
# TEXT CLEAN
# ===============================================================
STOPWORDS = {
    "dan","di","ke","yang","dari","untuk","dengan","adalah","ini",
    "itu","ada","atau","juga","sudah","akan","pada","dalam","tidak",
    "kami","kita","bisa","saya","anda","nya","bekasi","kota",
    "restoran","warung","kedai","makan","makanan","tempat","menu",
}

def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(t for t in text.split() if t not in STOPWORDS)


# ===============================================================
# MENU CONFIG
# ===============================================================
MENU_KATEGORI = {
    "jajanan":      "jajanan street food",
    "cafe":         "cafe dessert minuman kopi",
    "ayam":         "ayam bebek geprek",
    "bakso_mie":    "bakso mie bakmi",
    "nasi":         "nasi goreng nasi padang",
    "seafood":      "seafood ikan udang",
    "sate":         "sate",
    "burger_pizza": "burger pizza",
    "semua":        "",
}

MENU_HARGA = {
    "10-25":  "10rb - 25rb",
    "25-50":  "25rb - 50rb",
    "50-100": "50rb - 100rb",
    "100+":   ">100rb",
    "semua":  "",
}

KECAMATAN_LIST = [
    "Bekasi Barat","Bekasi Timur","Bekasi Selatan","Bekasi Utara",
    "Rawalumbu","Mustika Jaya","Bantargebang","Pondok Melati",
    "Pondok Gede","Jatisampurna","Jatiasih","Medan Satria",
]


# ===============================================================
# SCHEMA
# ===============================================================
class ChatRequest(BaseModel):
    step:      str = "start"
    pilihan:   str = ""
    kategori:  str = ""
    harga:     str = ""
    kecamatan: str = ""
    limit:     int = 6

class ChatResponse(BaseModel):
    step:             str
    pesan:            str
    tombol:           list
    hasil:            list = []
    rekomendasi_teks: list = []
    selesai:          bool = False


# ===============================================================
# HELPER: SAFE RECORD
# ===============================================================
def safe_records(dataframe):
    records = dataframe.replace({np.nan: None}).to_dict(orient="records")
    for r in records:
        r.setdefault("photo_url", None)
        r.setdefault("photo_source", None)

        if r["photo_url"] in ("", "nan", "None"):
            r["photo_url"] = None

        if r["photo_url"] and str(r["photo_url"]).startswith(("C:\\", "D:\\", "/")):
            fname = os.path.basename(str(r["photo_url"]))
            r["photo_url"] = f"https://api.jajanbekasi.web.id/foto/foto/{fname}"

    return records


# ===============================================================
# KNN RETRIEVAL
# Langkah 1: cari k kandidat terdekat menggunakan KNN
# Return: array index global kandidat + cosine distance-nya
# ===============================================================
def knn_retrieve(cbf_vec, k=30):
    """
    Gunakan KNN untuk mencari k item paling mirip secara konten.
    - cbf_vec : sparse matrix hasil vectorizer.transform()
    - k       : jumlah kandidat yang dikembalikan
    Return: (indices, distances) — keduanya array 1D
    """
    if knn_model is None:
        # Fallback jika model lama (belum include KNN)
        return None, None

    # Pastikan k tidak melebihi total data
    n_samples = tfidf_matrix.shape[0]
    k_actual  = min(k, n_samples)

    distances, indices = knn_model.kneighbors(cbf_vec, n_neighbors=k_actual)
    return indices.flatten(), distances.flatten()


# ===============================================================
# CBF RANKING
# Langkah 2: hitung final_score dari kandidat KNN
# ===============================================================
def rank_cbf(tmp, cbf_vec=None, query_text="", knn_indices=None):
    tmp = tmp.copy()

    if cbf_vec is not None:
        if knn_indices is not None:
            # ── Mode KNN: hitung similarity hanya pada kandidat KNN ──
            # Buat mapping global index → posisi lokal dalam tmp
            global_to_local = {
                global_idx: local_pos
                for local_pos, global_idx in enumerate(tmp.index)
            }

            sim_scores = np.zeros(len(tmp))
            for global_idx in knn_indices:
                if global_idx in global_to_local:
                    local_pos = global_to_local[global_idx]
                    # Hitung cosine similarity item kandidat vs query
                    score = cosine_similarity(
                        cbf_vec, tfidf_matrix[global_idx]
                    ).flatten()[0]
                    sim_scores[local_pos] = score

            tmp["cbf_score"] = sim_scores
        else:
            # ── Mode fallback: brute-force semua item ──
            sim = cosine_similarity(cbf_vec, tfidf_matrix[tmp.index]).flatten()
            tmp["cbf_score"] = sim
    else:
        tmp["cbf_score"] = 0.5

    # ── Bobot awal ──
    w_harga  = BOBOT_HARGA
    w_rating = BOBOT_RATING
    w_cbf    = BOBOT_CBF

    if query_text:
        query_lower = query_text.lower()

        # 1. Pisahkan location keyword dari food keyword
        found_kec        = set()
        found_kel        = set()
        location_keywords = set()

        for kel, kec in KECAMATAN_MAPPING.items():
            if kel in query_lower:
                found_kel.add(kel)
                location_keywords.update(kel.split())
            if kec.lower() in query_lower:
                found_kec.add(kec.lower())
                location_keywords.update(kec.lower().split())

        # 2. Food keyword exact boost
        exact_boost  = np.zeros(len(tmp))
        ignore_words = {
            "enak","murah","nyaman","halal","di","dan","atau",
            "dengan","daerah","wilayah","kota"
        }
        q_words = [
            w for w in query_lower.split()
            if len(w) > 2 and w not in ignore_words and w not in location_keywords
        ]
        food_query_str = " ".join(q_words)

        nama_lower     = tmp["nama"].astype(str).str.lower()
        kategori_lower = tmp["kategori"].astype(str).str.lower()

        if food_query_str:
            exact_boost += np.where(
                nama_lower.str.contains(food_query_str, regex=False, na=False),
                3.0, 0.0
            )

        if q_words:
            for w in q_words:
                exact_boost += np.where(
                    nama_lower.str.contains(w, regex=False, na=False), 0.5, 0.0
                )
                exact_boost += np.where(
                    kategori_lower.str.contains(w, regex=False, na=False), 0.3, 0.0
                )

        tmp["cbf_score"] = tmp["cbf_score"] + exact_boost

        # 3. Location multiplier
        loc_multiplier = np.ones(len(tmp))
        if found_kec or found_kel:
            kecamatan_lower = tmp["kecamatan"].astype(str).str.lower()
            address_lower   = tmp["address"].astype(str).str.lower() \
                              if "address" in tmp.columns else pd.Series([""] * len(tmp))

            is_loc_match = np.zeros(len(tmp), dtype=bool)
            for k in found_kec:
                is_loc_match |= kecamatan_lower.str.contains(k, regex=False, na=False)
            for k in found_kel:
                is_loc_match |= address_lower.str.contains(k, regex=False, na=False)

            loc_multiplier = np.where(is_loc_match, 1.5, 0.05)

        # 4. Dynamic weight berdasarkan intent
        boost_cbf    = 0.0
        boost_harga  = 0.0
        boost_rating = 0.0

        food_keywords = [
            "ayam","bebek","bakso","mie","kopi","cafe","nasi","padang",
            "seafood","jajanan","sate","soto","burger","pizza","kue","roti",
            "ikan","es ","minuman","geprek","bakar","goreng","sunda","warteg",
            "seblak","dimsum",
        ]

        if any(w in query_lower for w in food_keywords) or q_words:
            boost_cbf += 0.50

        if any(w in query_lower for w in ["murah","hemat","harga","10rb","25rb","50rb"]):
            boost_harga += 0.20

        if any(w in query_lower for w in ["enak","viral","hits","populer","terbaik","rating","nyaman","rekomendasi"]):
            boost_rating += 0.20

        # Normalisasi bobot
        total_w  = (w_cbf + boost_cbf) + (w_harga + boost_harga) + (w_rating + boost_rating)
        w_cbf    = (w_cbf    + boost_cbf)    / total_w
        w_harga  = (w_harga  + boost_harga)  / total_w
        w_rating = (w_rating + boost_rating) / total_w

        tmp["final_score"] = (
            w_harga  * tmp["price_score"] +
            w_rating * tmp["rating_norm"] +
            w_cbf    * tmp["cbf_score"]
        ) * loc_multiplier

    else:
        tmp["final_score"] = (
            w_harga  * tmp["price_score"] +
            w_rating * tmp["rating_norm"] +
            w_cbf    * tmp["cbf_score"]
        )

    return tmp


# ===============================================================
# REKOMENDASI (Chatbot — pakai KNN)
# ===============================================================
def rekomendasi(kategori, harga, kecamatan, limit):
    tmp = df.copy()

    query_parts = []
    if kategori and kategori != "semua":
        query_parts.append(MENU_KATEGORI.get(kategori, ""))
    if kecamatan and kecamatan != "semua":
        query_parts.append(kecamatan)
    query_parts.extend(["enak", "murah", "nyaman"])

    cbf_vec = vectorizer.transform([clean(" ".join(query_parts))])

    # ── Filter harga ──
    if harga and harga != "semua":
        val      = MENU_HARGA[harga].lower()
        filtered = tmp[tmp["price_label"].str.contains(val, na=False)]
        if len(filtered) > 0:
            tmp = filtered

    # ── Filter kecamatan ──
    if kecamatan and kecamatan != "semua":
        kec      = kecamatan.lower().strip()
        filtered = tmp[tmp["kecamatan"].astype(str).str.lower().str.contains(kec, na=False)]
        if len(filtered) > 0:
            tmp = filtered

    # ── KNN retrieval → CBF ranking ──
    knn_idx, _ = knn_retrieve(cbf_vec, k=50)    # ✅ Step 1: KNN cari kandidat
    tmp = rank_cbf(tmp, cbf_vec, " ".join(query_parts), knn_indices=knn_idx)  # ✅ Step 2: ranking

    hasil = tmp.sort_values("final_score", ascending=False).head(limit)
    return safe_records(hasil)


# ===============================================================
# API: POPULAR
# ===============================================================
@app.get("/api/popular")
def get_popular(limit: int = 8):
    tmp        = df.copy()
    query_text = "enak murah nyaman halal"
    cbf_vec    = vectorizer.transform([clean(query_text)])

    knn_idx, _ = knn_retrieve(cbf_vec, k=50)    # ✅ KNN
    tmp        = rank_cbf(tmp, cbf_vec, query_text, knn_indices=knn_idx)

    return safe_records(tmp.sort_values("final_score", ascending=False).head(limit))


# ===============================================================
# API: POPULAR BY KATEGORI (CBF + KNN)
# ===============================================================
@app.get("/api/popular/kategori")
def get_popular_by_kategori(limit_per_kategori: int = 4):
    KATEGORI_CBF = [
        ("Jajanan / Street Food", "jajanan street food gorengan cilok"),
        ("Cafe & Minuman",        "cafe kopi coffee minuman dessert boba aesthetic"),
        ("Ayam & Bebek",          "ayam bebek geprek goreng bakar"),
        ("Bakso & Mie",           "bakso mie bakmi pangsit ayam"),
        ("Nasi & Masakan",        "nasi goreng padang warteg sunda"),
        ("Seafood",               "seafood ikan udang cumi bakar"),
    ]

    result = {}
    for label, keywords in KATEGORI_CBF:
        cbf_vec    = vectorizer.transform([clean(keywords)])
        tmp        = df.copy()

        knn_idx, _ = knn_retrieve(cbf_vec, k=50)    # ✅ KNN
        tmp        = rank_cbf(tmp, cbf_vec, keywords, knn_indices=knn_idx)

        relevan = tmp[tmp["cbf_score"] > 0.05].sort_values(
            "final_score", ascending=False
        ).head(limit_per_kategori)

        if len(relevan) > 0:
            result[label] = safe_records(relevan)

    return result


# ===============================================================
# API: SEARCH (Cari Rekomendasi — pakai CBF + KNN)
#
# ✅ REVISI (fix "hasil rekomendasi dari klik kecamatan tidak muncul"):
#   SEBELUM: endpoint ini SELALU mewajibkan `q` diisi. Kalau user klik
#   kecamatan di homepage (tombol "Cari Berdasarkan Kecamatan"), request
#   yang terkirim adalah ?kecamatan=... TANPA `q`. Karena `q` kosong,
#   endpoint langsung return {error: "empty_query", data: []} — padahal
#   `kecamatan` sudah terisi dan datanya sebenarnya ada. Ini yang bikin
#   halaman Cari Rekomendasi tampil "Kuliner tidak ditemukan" walau
#   filter kecamatan sudah aktif dengan benar.
#
#   SESUDAH: keyword (`q`) ATAU kecamatan (`kecamatan`), salah satu
#   cukup untuk memproses request. Kalau `q` kosong tapi `kecamatan`
#   diisi, `query_text` fallback ke nama kecamatan + kata umum
#   ("enak murah nyaman") supaya CBF/KNN tetap py punya basis scoring,
#   bukan dianggap query kosong.
#
#   LOGIKA RANKING INTI (rank_cbf / knn_retrieve) TIDAK DIUBAH SAMA
#   SEKALI — hanya syarat wajib `q` dan cara membangun query_text yang
#   disesuaikan supaya kasus kecamatan-only tetap diproses dengan benar.
# ===============================================================
@app.get("/api/search")
def search(q: str = "", limit: int = 8, kecamatan: str = "", price: str = ""):

    has_query     = bool(q and q.strip())
    has_kecamatan = bool(kecamatan and kecamatan.strip() and kecamatan.lower() != "semua")

    # ── WAJIB ADA KEYWORD ATAU KECAMATAN ──
    # Cari Rekomendasi bukan listing biasa (itu tugas /api/daftar-kuliner),
    # tapi kalau user datang lewat klik kecamatan (tanpa keyword teks),
    # request tetap valid selama kecamatan-nya jelas.
    if not has_query and not has_kecamatan:
        return {
            "error": "empty_query",
            "message": "Masukkan kata kunci pencarian ya, misal 'ayam geprek murah'.",
            "data": []
        }

    tmp = df.copy()

    # ── Filter kecamatan ──
    if has_kecamatan:
        kec = kecamatan.lower().strip().replace("-", " ").replace("_", " ")
        tmp["kecamatan_clean"] = (
            tmp["kecamatan"].astype(str).str.lower().str.strip()
            .str.replace("-", " ", regex=False).str.replace("_", " ", regex=False)
        )
        filtered = tmp[tmp["kecamatan_clean"].str.contains(kec, na=False)]
        if len(filtered) > 0:
            tmp = filtered

    if len(tmp) == 0:
        return {"error": None, "message": None, "data": []}

    # ── Filter harga ──
    if price and price.strip() not in ("semua", ""):
        price_val    = price.lower().strip()
        price_mapped = MENU_HARGA.get(price_val, price_val).lower()
        if "100" in price_val and ">" in price_val:
            price_mapped = ">100rb"
        price_filtered = tmp[tmp["price_label"].str.lower().str.contains(
            price_mapped.replace(">", "\\>"), na=False, regex=True
        )]
        if len(price_filtered) > 0:
            tmp = price_filtered

    # ── Build query text (+ kelurahan terms) ──
    # Kalau q kosong (murni klik kecamatan tanpa keyword), fallback ke
    # nama kecamatan itu sendiri supaya CBF/KNN tetap punya basis teks.
    query_text     = q.strip() if has_query else kecamatan.strip()
    kelurahan_terms = []

    if has_kecamatan:
        kec_lower = kecamatan.lower().strip()
        for kel, kec_val in KECAMATAN_MAPPING.items():
            if kec_val.lower() == kec_lower:
                kelurahan_terms.append(kel)

    q_lower = query_text.lower()
    for kel, kec_val in KECAMATAN_MAPPING.items():
        if kec_val.lower() in q_lower:
            kelurahan_terms.append(kel)

    if kelurahan_terms:
        query_text = (query_text + " " + " ".join(set(kelurahan_terms))).strip()

    if not has_query:
        # Tambahkan kata umum biar CBF tidak terlalu sempit kalau cuma
        # ada nama kecamatan tanpa keyword makanan apapun sama sekali.
        query_text += " enak murah nyaman"

    # ── KNN retrieval → CBF ranking (LOGIKA ASLI, TIDAK DIUBAH) ──
    cbf_vec    = vectorizer.transform([clean(query_text)])
    knn_idx, _ = knn_retrieve(cbf_vec, k=50)
    tmp        = rank_cbf(tmp, cbf_vec, query_text, knn_indices=knn_idx)

    # ── FILTER RELEVANSI ──
    # Sebelumnya endpoint ini selalu ambil top-N (head(limit)) dari SELURUH
    # data yang sudah difilter kecamatan/harga, walau cbf_score-nya sangat
    # rendah / tidak nyambung sama sekali dengan keyword. Akibatnya jumlah
    # hasil terlihat "dipaksa" sampai 24 walau yang relevan cuma segelintir.
    #
    # FIX: kalau user memasukkan keyword (has_query), buang kandidat yang
    # cbf_score-nya di bawah ambang relevansi. Kalau cuma filter kecamatan
    # tanpa keyword, tidak ada ambang tambahan (semua kuliner di kecamatan
    # itu tetap relevan secara lokasi).
    #
    # LOGIKA RANKING INTI (rank_cbf / knn_retrieve / bobot) TIDAK DIUBAH —
    # ini cuma memotong hasil yang tidak relevan sebelum head(limit).
    RELEVANCE_THRESHOLD = 0.05
    if has_query:
        relevan = tmp[tmp["cbf_score"] > RELEVANCE_THRESHOLD]
        tmp = relevan if len(relevan) > 0 else tmp.iloc[0:0]

    hasil = tmp.sort_values("final_score", ascending=False).head(limit)

    return {
        "error": None,
        "message": None,
        "data": safe_records(hasil)
    }


# ===============================================================
# API: DAFTAR KULINER (browsing semua data, TANPA scoring TF-IDF/KNN)
# Search di sini cuma simple text-match di kolom nama.
# Endpoint ini TERPISAH dari sistem rekomendasi — tidak menyentuh
# rank_cbf / knn_retrieve sama sekali.
#
# ✅ UPDATE:
#   - filter kategori (contains, case-insensitive)
#   - filter kecamatan (exact match, sesuai nilai baku di KECAMATAN_MAPPING)
#   - sort: rating_desc (default) | rating_asc | name_asc | price_asc | price_desc
#   - stats: dihitung dari hasil SETELAH difilter (q/kategori/kecamatan),
#            SEBELUM pagination — akurat untuk seluruh hasil pencarian,
#            bukan cuma yang tampil di 1 halaman.
# ===============================================================

# Urutan harga termurah -> termahal, dipakai untuk sort price_asc/price_desc.
PRICE_ORDER = {
    "10rb - 25rb":  1,
    "25rb - 50rb":  2,
    "50rb - 100rb": 3,
    ">100rb":       4,
}

@app.get("/api/daftar-kuliner")
def daftar_kuliner(
    q: str = "",               # opsional: cari nama secara simple text match
    kategori: str = "",        # opsional: filter kategori (contains)
    kecamatan: str = "",       # opsional: filter kecamatan (exact)
    sort: str = "rating_desc", # rating_desc | rating_asc | name_asc | price_asc | price_desc
    page: int = 1,
    limit: int = 40,           # default 40 per halaman
):
    # ── Cap limit maksimal 100 per request ──
    limit = max(1, min(limit, 100))

    tmp = df.copy()

    # ── Simple text search di nama (BUKAN TF-IDF) ──
    if q and q.strip():
        q_clean = q.strip().lower()
        tmp = tmp[tmp["nama"].astype(str).str.lower().str.contains(q_clean, na=False)]

    # ── Filter kategori ──
    if kategori and kategori.strip() and kategori.lower() != "semua":
        kat_clean = kategori.strip().lower()
        tmp = tmp[tmp["kategori"].astype(str).str.lower().str.contains(kat_clean, na=False)]

    # ── Filter kecamatan ──
    if kecamatan and kecamatan.strip() and kecamatan.lower() != "semua":
        kec_clean = kecamatan.strip().lower()
        tmp = tmp[tmp["kecamatan"].astype(str).str.lower().str.strip() == kec_clean]

    # ── STATS: dihitung dari hasil filter (sebelum pagination) ──
    if len(tmp) > 0:
        rating_avg_val = float(tmp["rating_avg"].mean()) if "rating_avg" in tmp.columns else None
    else:
        rating_avg_val = None

    stats = {
        "total_kuliner":   int(len(tmp)),
        "total_kecamatan": int(tmp["kecamatan"].nunique()) if len(tmp) > 0 else 0,
        "rating_avg":      round(rating_avg_val, 1) if rating_avg_val is not None else None,
        "total_kategori":  int(tmp["kategori"].nunique()) if len(tmp) > 0 else 0,
    }

    total       = len(tmp)
    total_pages = max(1, (total + limit - 1) // limit)
    page        = max(1, min(page, total_pages))

    # ── Sorting ──
    if sort == "rating_asc":
        tmp = tmp.sort_values("rating_avg", ascending=True, na_position="last")
    elif sort == "name_asc":
        tmp = tmp.sort_values("nama", ascending=True, na_position="last")
    elif sort == "price_asc" or sort == "price_desc":
        tmp = tmp.copy()
        tmp["_price_order"] = tmp["price_label"].map(PRICE_ORDER).fillna(99)
        tmp = tmp.sort_values(
            "_price_order",
            ascending=(sort == "price_asc"),
            na_position="last"
        )
        tmp = tmp.drop(columns=["_price_order"])
    else:
        # default: rating_desc (perilaku asli tetap dipertahankan)
        tmp = tmp.sort_values("rating_avg", ascending=False, na_position="last")

    start = (page - 1) * limit
    end   = start + limit
    hasil = tmp.iloc[start:end]

    return {
        "total":       total,
        "page":        page,
        "total_pages": total_pages,
        "limit":       limit,
        "stats":       stats,
        "data":        safe_records(hasil),
    }


# ===============================================================
# API: TRENDING PER KECAMATAN
# Ambil top-6 (rating tertinggi) untuk masing-masing 12 kecamatan.
# Dipakai di halaman Cari Rekomendasi. Murni sorting rating,
# tidak menyentuh sistem rekomendasi CBF/KNN.
# ===============================================================
@app.get("/api/trending-kecamatan")
def trending_kecamatan(limit_per_kecamatan: int = 6):
    result = {}

    for kec in KECAMATAN_LIST:
        tmp = df[
            df["kecamatan"].astype(str).str.lower().str.strip() == kec.lower()
        ].copy()

        if len(tmp) == 0:
            continue

        tmp = tmp.sort_values("rating_avg", ascending=False).head(limit_per_kecamatan)
        result[kec] = safe_records(tmp)

    return result


# ===============================================================
# API: KEYWORD SUGGESTIONS
# ===============================================================
@app.get("/api/keywords")
def get_keywords():
    return {
        "popular_keywords": [
            "makanan murah bekasi", "cafe aesthetic bekasi", "ayam geprek enak",
            "bakso enak bekasi", "tempat makan nyaman", "kuliner malam bekasi",
            "jajanan street food", "cafe kopi murah", "nasi padang bekasi",
            "makanan halal bekasi", "seafood bekasi", "tempat nongkrong viral",
            "sarapan murah bekasi", "dessert enak bekasi", "mie ayam enak",
            "ayam bakar bekasi", "bebek goreng gurih", "sate madura bekasi",
            "sate padang enak", "bakso mercon pedas", "mie ayam pangsit",
            "mie gacor pedas", "seblak prasmanan bekasi", "dimsum hangat murah",
            "martabak manis cokelat", "martabak telur spesial", "pempek palembang bekasi",
            "seafood kiloan segar", "ikan bakar bekasi", "nasi goreng kambing",
            "bubur ayam bekasi", "soto betawi kuah santan", "soto lamongan asli",
            "kuliner galaxy bekasi", "kuliner summarecon bekasi", "cafe instagramable",
            "tempat nongkrong 24 jam", "kuliner legendaris bekasi", "makanan ramah anak",
            "angkringan malam bekasi", "kopi susu kekinian", "warteg bersih murah",
            "steamboat bekasi", "restoran sunda bekasi", "ramen enak bekasi",
            "gelato bekasi", "roti bakar bekasi", "mie bakso sapi",
            "kuliner bekasi selatan", "kuliner bekasi timur", "kuliner bekasi barat",
            "kuliner bekasi utara", "kuliner rawalumbu", "kuliner jatiasih",
            "kuliner pondok gede", "kuliner medan satria", "kuliner mustika jaya",
            "kuliner jatisampurna",
        ],
        "quick_search": [
            {"label": "🔥 Viral & Hits",   "q": "viral hits bekasi"},
            {"label": "💰 Murah < 25rb",   "q": "murah ekonomis"},
            {"label": "☕ Cafe Aesthetic",  "q": "cafe aesthetic cozy"},
            {"label": "🍗 Ayam Geprek",    "q": "ayam geprek pedas"},
            {"label": "🍜 Bakso & Mie",    "q": "bakso mie enak"},
            {"label": "🌙 Kuliner Malam",  "q": "kuliner malam buka"},
            {"label": "🐟 Seafood",        "q": "seafood ikan udang"},
            {"label": "🍱 Nasi Padang",    "q": "nasi padang masakan minang"},
        ],
    }


# ===============================================================
# API: FOTO
# ===============================================================
@app.get("/api/foto")
def get_foto(nama: str):
    tmp = df.copy()
    tmp["nama_lower"] = tmp["nama"].astype(str).str.lower()
    hasil = tmp[tmp["nama_lower"].str.contains(nama.lower().strip(), na=False)]
    if hasil.empty:
        return {"photo_url": None, "photo_source": None}
    row = hasil.iloc[0]
    return {
        "photo_url":    row.get("photo_url") or None,
        "photo_source": row.get("photo_source") or None,
    }


# ===============================================================
# CHATBOT
# ===============================================================
def build_rekomendasi_teks(records):
    if not records:
        return []

    def fmt(r, n):
        nama      = r.get('nama') or '-'
        kecamatan = r.get('kecamatan') or '-'
        kategori  = r.get('kategori') or '-'
        harga     = r.get('price_label') or 'bervariasi'
        rating    = float(r.get('rating_avg') or 4.0)

        templates = [
            f"**#{n} {nama}**, {kecamatan} — Kategori {kategori}, harga {harga}, rating {rating:.1f}/5.",
            f"**{nama}** ({kategori}) berlokasi di {kecamatan}, patut dicoba dengan rating {rating:.1f}/5.",
            f"**{nama}**, {kecamatan} — menyajikan hidangan lezat dengan harga {harga} (Rating: {rating:.1f}/5).",
            f"**{nama}**, {kecamatan} — Pilihan menarik untuk {kategori} (Rating: {rating:.1f}/5).",
            f"**{nama}**, {kecamatan} — rating {rating:.1f}/5, harga {harga}.",
        ]
        return templates[n - 1] if n <= len(templates) else templates[0]

    return [fmt(item, idx + 1) for idx, item in enumerate(records[:5])]

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    if req.step == "start":
        user_name = req.user_name.strip() if getattr(req, 'user_name', None) and req.user_name not in ('null', 'undefined') else ""
        greeting = f"Halo {user_name}! Saya JajanAI Mau makan apa hari ini? Ceritain aja kamu lagi pengen makanan apa, di area mana, atau budget berapa — saya siap kasih rekomendasi terbaik kuliner Bekasi!" if user_name else "Halo! Saya JajanAI. mau makan apa hari ini? Ceritain aja kamu lagi pengen makanan apa, di area mana, atau budget berapa saya siap kasih rekomendasi terbaik kuliner Bekasi!"
        return ChatResponse(
            step="kategori",
            pesan=greeting,
            tombol=[{"label": k.replace("_", " ").title(), "value": k} for k in MENU_KATEGORI],
        )

    if req.step == "kategori":
        return ChatResponse(
            step="harga",
            pesan="💰 Pilih kisaran harga:",
            kategori=req.pilihan,
            tombol=[
                {"label": "< 25rb (paling favorit)",   "value": "10-25"},
                {"label": "25rb – 50rb ⭐ terbanyak",  "value": "25-50"},
                {"label": "50rb – 100rb",               "value": "50-100"},
                {"label": "> 100rb",                    "value": "100+"},
                {"label": "Semua harga",                "value": "semua"},
            ],
        )

    if req.step == "harga":
        return ChatResponse(
            step="kecamatan",
            pesan="📍 Pilih lokasi kecamatan:",
            kategori=req.kategori,
            harga=req.pilihan,
            tombol=[{"label": k, "value": k} for k in KECAMATAN_LIST]
            + [{"label": "Semua lokasi", "value": "semua"}],
        )

    if req.step == "kecamatan":
        # Force limit to 5
        hasil = rekomendasi(req.kategori, req.harga, req.pilihan, 5)
        rekomendasi_teks = build_rekomendasi_teks(hasil)
        
        lokasi = req.pilihan if req.pilihan.lower() != "semua" else "Bekasi"
        kategori_label = req.kategori.replace("_", " ").title() if req.kategori else "kuliner"
        
        if len(hasil) == 0:
            pesan = f"Aduh, maaf sekali. Di {lokasi} saat ini belum ada data untuk kategori {kategori_label}. Coba cari yang lain ya! 🥺"
        else:
            pesan = f"Wah, pilihan yang mantap! Di {lokasi} ada beberapa rekomendasi {kategori_label} yang terkenal enak. Coba intip rekomendasinya di bawah ini ya! 🍜"
            
        return ChatResponse(
            step="hasil",
            pesan=pesan,
            hasil=hasil,
            rekomendasi_teks=rekomendasi_teks,
            tombol=[{"label": "🔄 Cari Lagi", "value": "start"}],
            selesai=True,
        )

    if req.step == "free_text":
        q          = clean(req.pilihan or "")
        cbf_vec    = vectorizer.transform([q])
        tmp        = df.copy()

        knn_idx, _ = knn_retrieve(cbf_vec, k=50)        # ✅ KNN
        tmp        = rank_cbf(tmp, cbf_vec, req.pilihan or "", knn_indices=knn_idx)

        # Force limit to 5
        hasil = tmp.sort_values("final_score", ascending=False).head(5)
        records = safe_records(hasil)
        rekomendasi_teks = build_rekomendasi_teks(records)
        
        if len(records) == 0:
            pesan = f"Hmm, JajanAI belum berhasil menemukan kuliner untuk '{req.pilihan}'. Coba masukkan kata kunci yang berbeda? 🥺"
        else:
            pesan = f"Wah, pilihan yang mantap! JajanAI berhasil menemukan beberapa kuliner terbaik yang paling cocok dengan '{req.pilihan}' di Bekasi. Ini rekomendasinya untuk kamu! 🍜"
            
        return ChatResponse(
            step="hasil",
            pesan=pesan,
            hasil=records,
            rekomendasi_teks=rekomendasi_teks,
            tombol=[{"label": "🔄 Cari Lagi", "value": "start"}],
            selesai=True,
        )

    return ChatResponse(step="start", pesan="Error", tombol=[])


# ===============================================================
# STARTUP
# ===============================================================
@app.on_event("startup")
async def startup_event():
    init_model()


# ===============================================================
# STATIC: FOTO
# ===============================================================
if os.path.exists(FOTO_DIR):
    app.mount("/foto", StaticFiles(directory=FOTO_DIR), name="foto")
else:
    print("⚠️ Folder foto/ belum ada")


# ===============================================================
# RUN
# ===============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True)