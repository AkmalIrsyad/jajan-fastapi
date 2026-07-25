# -*- coding: utf-8 -*-

import os          # <- TAMBAHKAN BARIS INI
import kagglehub
import pandas as pd
import numpy as np
import ast
import re

# =========================================================
# DOWNLOAD DATASET
# =========================================================
path = kagglehub.dataset_download("iannarsa/gofood-merchant-on-jabodetabek-and-bandung")
print("Dataset path:", path)

# <- UBAH BARIS INI UNTUK MENGGABUNGKAN PATH DENGAN NAMA FILE
INPUT_FILE = os.path.join(path, "gofood_Jabodetabek_bandung.csv") 
KUESIONER_FILE = "ResponsKeywordAnalysis.csv"

# =========================================================
# LOAD DATA
# =========================================================
df_raw = pd.read_csv(INPUT_FILE, low_memory=False)

# =========================================================
# FILTER BEKASI
# =========================================================
df_bekasi = df_raw[df_raw["areaName"].str.contains("bekasi", case=False, na=False)].copy()

# =========================================================
# FUNCTIONS
# =========================================================
def parse_rating(raw):
    try:
        d = ast.literal_eval(str(raw))
        if isinstance(d, dict):
            return float(d.get("average", 0)), int(d.get("total", 0))
    except:
        pass
    return 0.0, 0

def parse_tags(raw):
    try:
        tags_list = ast.literal_eval(str(raw))
        if isinstance(tags_list, list):
            return ", ".join(
                t.get("displayName", "")
                for t in tags_list
                if isinstance(t, dict)
            )
    except:
        pass
    return ""

# =========================================================
# ✅ REVISI 1: KECAMATAN MAPPING (kelurahan → kecamatan)
# -----------------------------------------------------------------
# SEBELUM (BUG): extract_kecamatan() hanya mengecek apakah salah satu
# dari 12 NAMA KECAMATAN muncul literal di areaName. Padahal areaName
# dari data GoFood umumnya berisi nama KELURAHAN (mis. "Cikiwul",
# "Sumur Batu", "Ciketing Udik" — semuanya bagian dari Bantargebang),
# bukan nama kecamatan induknya. Akibatnya baris-baris tsb gagal
# ter-mapping, kecamatan-nya jadi "", lalu DIBUANG oleh baris:
#     df_bekasi = df_bekasi[df_bekasi["kecamatan"] != ""]
# Data hilang permanen sebelum sempat disimpan ke CSV / dipakai model.
#
# SESUDAH (FIX): pakai mapping kelurahan → kecamatan yang sama seperti
# di model_training.py dan app.py, supaya satu sumber kebenaran dan
# semua kelurahan berhasil ter-resolve ke kecamatan induknya.
#
# ✅ REVISI 2 (tambahan setelah audit CSV hasil generate):
# Ditemukan baris seperti "Pecel Lele Pak Kumis, Bantargebang" dan
# "Bakso RT Pangkalan 1B, Bantar Gebang" yang address-nya JELAS
# menyebut Bantargebang, tapi kolom areaName mereka gagal terdeteksi
# (kecamatan jadi NaN/kosong lalu ikut kebuang). Ditambahkan alias
# "bantar gebang" (dengan spasi) karena variasi penulisan ini belum
# ada di mapping sebelumnya.
# =========================================================
KECAMATAN_MAPPING = {
    # Bekasi Barat
    "bintara":            "Bekasi Barat",
    "bintara jaya":       "Bekasi Barat",
    "bintarajaya":        "Bekasi Barat",
    "jakasampurna":       "Bekasi Barat",
    "kranji":             "Bekasi Barat",
    "kota baru":          "Bekasi Barat",
    "kotabaru":           "Bekasi Barat",

    # Bekasi Timur
    "aren jaya":          "Bekasi Timur",
    "arenjaya":           "Bekasi Timur",
    "bekasi jaya":        "Bekasi Timur",
    "bekasijaya":         "Bekasi Timur",
    "duren jaya":         "Bekasi Timur",
    "durenjaya":          "Bekasi Timur",
    "margahayu":          "Bekasi Timur",
    "rawasemut":          "Bekasi Timur",
    "bulak kapal":        "Bekasi Timur",
    "bulakkapal":         "Bekasi Timur",

    # Bekasi Selatan
    "jakamulya":          "Bekasi Selatan",
    "jakasetia":          "Bekasi Selatan",
    "kayuringin jaya":    "Bekasi Selatan",
    "kayuringinjaya":     "Bekasi Selatan",
    "marga jaya":         "Bekasi Selatan",
    "margajaya":          "Bekasi Selatan",
    "pekayon":            "Bekasi Selatan",
    "pekayon jaya":       "Bekasi Selatan",
    "pekayonjaya":        "Bekasi Selatan",

    # Bekasi Utara
    "harapan baru":       "Bekasi Utara",
    "harapanbaru":        "Bekasi Utara",
    "harapan jaya":       "Bekasi Utara",
    "harapanjaya":        "Bekasi Utara",
    "kaliabang tengah":   "Bekasi Utara",
    "kaliabangtengah":    "Bekasi Utara",
    "marga mulya":        "Bekasi Utara",
    "margamulya":         "Bekasi Utara",
    "perwira":            "Bekasi Utara",
    "teluk pucung":       "Bekasi Utara",
    "telukpucung":        "Bekasi Utara",
    "summarecon bekasi":  "Bekasi Utara",
    "pondok ungu permai":  "Bekasi Utara",

    # Rawalumbu
    "bojong rawalumbu":   "Rawalumbu",
    "bojongrawalumbu":    "Rawalumbu",
    "bojong menteng":     "Rawalumbu",
    "bojongmenteng":      "Rawalumbu",
    "pengasinan":         "Rawalumbu",
    "sepanjangjaya":      "Rawalumbu",
    "sepanjang jaya":     "Rawalumbu",
    "narogong":           "Rawalumbu",
    "pondok hijau":       "Rawalumbu",

    # Mustika Jaya
    "cimuning":           "Mustika Jaya",
    "mustikajaya":        "Mustika Jaya",
    "mustika jaya":       "Mustika Jaya",
    "mustikasari":        "Mustika Jaya",
    "pedurenan":          "Mustika Jaya",
    "padurenan":          "Mustika Jaya",
    "mutiara gading":     "Mustika Jaya",

    # Bantargebang
    "bantargebang":       "Bantargebang",
    "bantar gebang":      "Bantargebang",   # ✅ alias dengan spasi (baru)
    "ciketing udik":      "Bantargebang",
    "ciketingudik":       "Bantargebang",
    "cikiwul":            "Bantargebang",
    "sumur batu":         "Bantargebang",
    "sumurbatu":          "Bantargebang",

    # Pondok Melati
    "pondok melati":      "Pondok Melati",
    "pondokmelati":       "Pondok Melati",
    "jatimelati":         "Pondok Melati",
    "jatimurni":          "Pondok Melati",
    "jatirahayu":         "Pondok Melati",
    "jatiwarna":          "Pondok Melati",

    # Pondok Gede
    "pondok gede":        "Pondok Gede",
    "pondokgede":         "Pondok Gede",
    "jatibening":         "Pondok Gede",
    "jatibening baru":    "Pondok Gede",
    "jatibeningbaru":     "Pondok Gede",
    "jaticempaka":        "Pondok Gede",
    "jatimakmur":         "Pondok Gede",
    "jatiwaringin":       "Pondok Gede",

    # Jatisampurna
    "jatisampurna":       "Jatisampurna",
    "jatikarya":          "Jatisampurna",
    "jatiraden":          "Jatisampurna",
    "jatirangga":         "Jatisampurna",
    "jatiranggon":        "Jatisampurna",
    "harjamukti":         "Jatisampurna",

    # Jatiasih
    "jatiasih":           "Jatiasih",
    "jatikramat":         "Jatiasih",
    "jatiluhur":          "Jatiasih",
    "jatimekar":          "Jatiasih",
    "jatirasa":           "Jatiasih",
    "jatisari":           "Jatiasih",

    # Medan Satria
    "harapan mulya":      "Medan Satria",
    "harapanmulya":       "Medan Satria",
    "kalibaru":           "Medan Satria",
    "kali baru":          "Medan Satria",
    "medansatria":        "Medan Satria",
    "medan satria":       "Medan Satria",
    "pejuang":            "Medan Satria",
    "kota harapan indah": "Medan Satria",
}


# =========================================================
# ✅ REVISI: EXTRACT KECAMATAN
# -----------------------------------------------------------------
# Sekarang mengecek KECAMATAN_MAPPING (kelurahan → kecamatan) alih-alih
# hanya mencocokkan 12 nama kecamatan besar secara literal.
#
# Catatan urutan pengecekan:
#   1. Cek dulu apakah salah satu KELURAHAN pada mapping muncul di teks.
#   2. Kalau tidak ketemu kelurahan spesifik, baru cek apakah nama
#      KECAMATAN besar itu sendiri muncul literal (fallback, untuk
#      kasus teks yang memang langsung menyebut nama kecamatan).
#   Key kelurahan dicek dari yang paling panjang ke pendek supaya
#   pencocokan substring lebih akurat (mis. "bojong rawalumbu" dicek
#   sebelum "rawalumbu").
# =========================================================
KECAMATAN_BESAR = [
    "bekasi barat", "bekasi timur", "bekasi selatan", "bekasi utara",
    "rawalumbu", "mustika jaya", "bantargebang", "bantar gebang",
    "pondok melati", "pondok gede", "jatisampurna", "jatiasih",
    "medan satria",
]

_KELURAHAN_SORTED = sorted(KECAMATAN_MAPPING.keys(), key=len, reverse=True)


def extract_kecamatan(text_raw):
    """
    Dipakai untuk areaName (biasanya HANYA berisi nama kelurahan/kecamatan
    saja, bukan alamat jalan lengkap) — jadi substring match sederhana
    sudah aman dipakai di sini.
    """
    text = str(text_raw).lower()

    # 1) Cek kelurahan spesifik dulu (lebih akurat & granular)
    for kel in _KELURAHAN_SORTED:
        if kel in text:
            return KECAMATAN_MAPPING[kel]

    # 2) Fallback: cek nama kecamatan besar langsung disebut di teks
    for kec in KECAMATAN_BESAR:
        if kec in text:
            return KECAMATAN_MAPPING.get(kec, kec.title())

    return ""


def extract_kecamatan_from_address(address_raw):
    """
    ✅ Khusus untuk kolom `address` (alamat lengkap, bukan cuma nama area).

    PENTING: substring match naif SALAH KAPRAH di sini, karena alamat
    lengkap sering menyebut nama JALAN yang kebetulan sama dengan nama
    kecamatan (mis. "Jalan Setu - Bantar Gebang" adalah nama jalan yang
    melintasi wilayah Mustika Jaya, BUKAN berarti lokasinya di kecamatan
    Bantargebang). Kalau langsung di-substring-match, kasus seperti ini
    salah tangkap nama jalan sebagai kelurahan.

    FIX: pecah address per koma (format umum: "Jalan X, Komplek Y,
    Kelurahan, Kota Bekasi, Jawa Barat, ..."), lalu cek tiap segmen
    SATU PER SATU mulai dari segmen paling BELAKANG (dekat "Kota
    Bekasi") ke depan — karena nama kelurahan asli biasanya ada di
    posisi itu, sedangkan nama jalan ada di segmen paling awal.
    Segmen pertama yang match dengan KECAMATAN_MAPPING langsung dipakai.
    """
    text = str(address_raw).lower()
    segments = [s.strip() for s in text.split(",") if s.strip()]

    # Scan dari belakang (paling dekat "kota bekasi") ke depan
    for seg in reversed(segments):
        for kel in _KELURAHAN_SORTED:
            if kel == seg or kel in seg:
                return KECAMATAN_MAPPING[kel]
        for kec in KECAMATAN_BESAR:
            if kec == seg or kec in seg:
                return KECAMATAN_MAPPING.get(kec, kec.title())

    return ""


def classify_area(address):
    addr = str(address).lower()
    if "kota bekasi" in addr:
        return "Kota Bekasi"
    elif "kabupaten bekasi" in addr:
        return "Kabupaten Bekasi"
    return "Bekasi"

# =========================================================
# PRICE MAP
# =========================================================
PRICE_MAP = {
    1: "10rb - 25rb",
    2: "25rb - 50rb",
    3: "50rb - 100rb",
    4: ">100rb"
}

def map_price(level):
    return PRICE_MAP.get(level, "25rb - 50rb")

# =========================================================
# CLEANING
# =========================================================
df_bekasi[["rating_avg", "rating_total"]] = df_bekasi["ratings"].apply(
    lambda x: pd.Series(parse_rating(x))
)

df_bekasi["kategori"] = df_bekasi["tags"].apply(parse_tags)

df_bekasi = df_bekasi.rename(columns={
    "displayName": "nama",
    "description": "deskripsi",
    "priceLevel": "price_level"
})

df_bekasi["price_label"] = df_bekasi["price_level"].apply(map_price)


# =========================================================
# ✅ REVISI 2: AMBIL KECAMATAN — DENGAN FALLBACK KE `address`
# -----------------------------------------------------------------
# TEMUAN AUDIT: setelah CSV hasil generate dicek langsung, ada
# beberapa kuliner yang address-nya JELAS menyebut nama kecamatan/
# kelurahan (mis. "Bantargebang, Kota Bekasi, ...") tapi kolom
# areaName mereka tidak cukup spesifik sehingga extract_kecamatan()
# gagal mendeteksi apa pun dari areaName saja → kecamatan jadi "".
#
# FIX: deteksi kecamatan dilakukan 2 tahap —
#   1. Coba deteksi dari `areaName` dulu (seperti sebelumnya).
#   2. Kalau hasil dari areaName kosong, coba lagi dari `address`
#      sebagai fallback, karena address biasanya lebih deskriptif
#      dan sering menyebut nama kelurahan/kecamatan secara eksplisit.
#
# Ini berlaku bukan cuma untuk Bantargebang saja, tapi otomatis
# menyelamatkan baris kecamatan manapun yang sebelumnya kebuang
# karena areaName-nya kurang informatif.
# =========================================================
kec_from_area = df_bekasi["areaName"].astype(str).apply(extract_kecamatan)

if "address" in df_bekasi.columns:
    kec_from_address = df_bekasi["address"].astype(str).apply(extract_kecamatan_from_address)
else:
    kec_from_address = pd.Series([""] * len(df_bekasi), index=df_bekasi.index)

# Pakai hasil dari areaName; kalau kosong, isi dari address (fallback)
df_bekasi["kecamatan"] = kec_from_area.where(kec_from_area != "", kec_from_address)

# Debug: berapa baris yang terselamatkan berkat fallback address
_selamat_dari_address = ((kec_from_area == "") & (kec_from_address != "")).sum()
print(f"\n📍 Baris terselamatkan lewat fallback address: {_selamat_dari_address}")

# hapus data yang kecamatannya tetap kosong setelah kedua tahap
df_bekasi = df_bekasi[df_bekasi["kecamatan"] != ""]

print("\n=== DATA KECAMATAN ===")
print(df_bekasi["kecamatan"].value_counts())

df_bekasi["area_type"] = df_bekasi["areaName"].apply(classify_area)

# FILTER ONLY KOTA BEKASI
df_bekasi = df_bekasi[df_bekasi["area_type"] == "Kota Bekasi"]

# =========================================================
# KUESIONER
# =========================================================
try:
    df_q = pd.read_csv(KUESIONER_FILE)
except:
    df_q = pd.DataFrame()

df_kuliner_q = pd.DataFrame()

if not df_q.empty:
    kolom = None
    for k in ["nama_kuliner","kuliner","makanan_favorit","rekomendasi"]:
        if k in df_q.columns:
            kolom = k
            break

    if kolom:
        df_kuliner_q = df_q[[kolom]].rename(columns={kolom:"nama"})
        df_kuliner_q["nama"] = df_kuliner_q["nama"].astype(str).str.strip()
        df_kuliner_q = df_kuliner_q[df_kuliner_q["nama"].str.len() > 2]

        df_kuliner_q["kategori"] = "Kuesioner"
        df_kuliner_q["kecamatan"] = ""
        df_kuliner_q["price_label"] = "25rb - 50rb"
        df_kuliner_q["rating_avg"] = 4.0
        df_kuliner_q["deskripsi"] = "Data kuesioner"
        df_kuliner_q["shortLink"] = ""
        df_kuliner_q["is_kuesioner"] = 1

# =========================================================
# GABUNG DATA
# =========================================================
df_bekasi["is_kuesioner"] = 0

KOLOM = [
    "nama","kategori","kecamatan","price_label",
    "rating_avg","deskripsi","shortLink","is_kuesioner"
]

df_gofood = df_bekasi.reindex(columns=KOLOM, fill_value="")
df_kuliner_q = df_kuliner_q.reindex(columns=KOLOM, fill_value="")

df = pd.concat([df_gofood, df_kuliner_q], ignore_index=True)

df["nama"] = df["nama"].astype(str)
df = df[df["nama"].str.len() > 3]

# DEBUG CEK KECAMATAN
print("\n=== SAMPLE KECAMATAN ===")
print(df["kecamatan"].value_counts().head(10))

# =========================================================
# SAVE FINAL DATASET
# =========================================================
# =========================================================
# GABUNG FOTO
# =========================================================
try:
    df_foto = pd.read_csv("bekasi_kuliner_with_photos.csv", usecols=["displayName", "photo_url", "photo_source"])
    df_foto = df_foto.rename(columns={"displayName": "nama"})
    df_foto["nama"] = df_foto["nama"].astype(str).str.strip()

    # normalisasi nama untuk matching
    df["nama_key"] = df["nama"].str.lower().str.strip()
    df_foto["nama_key"] = df_foto["nama"].str.lower().str.strip()

    df = df.merge(df_foto[["nama_key","photo_url","photo_source"]], on="nama_key", how="left")
    df = df.drop(columns=["nama_key"])

    print(f"✅ Foto berhasil digabung: {df['photo_url'].notna().sum()} dari {len(df)} data punya foto")
except Exception as e:
    df["photo_url"] = ""
    df["photo_source"] = ""
    print(f"⚠️ Foto tidak digabung: {e}")

# =========================================================
# SAVE FINAL DATASET
# =========================================================
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gofood_bekasi_kota.csv")

df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ Dataset saved:", OUTPUT_FILE)
print("Total data:", len(df))
print(df["kecamatan"].unique()[:20])