# -*- coding: utf-8 -*-
"""
foto_scrapper.py
=================
Ambil foto kuliner yang BELUM ada fotonya dari dataset Bekasi.
- GoFood OG Image (utama)
- Bing Image Search (fallback, menggantikan DDG yang diblokir)
- Skip file yang sudah ada di disk (deteksi dengan/tanpa prefix angka)
- Append ke bekasi_kuliner_with_photos.csv tanpa duplikasi

Target: total 800 foto dari 829 data.

Cara pakai:
    pip install requests pandas
    python foto_scrapper.py
"""

import os
import re
import time
import ast
import requests
import pandas as pd

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
INPUT_CSV    = r"D:\laragon\www\JajanBekasi\data\bekasi_kota_only.csv"
OUTPUT_CSV   = r"D:\laragon\www\JajanBekasi\data\bekasi_kuliner_with_photos.csv"
FOTO_DIR     = r"D:\laragon\www\JajanBekasi\foto"
FASTAPI_BASE = "http://localhost:8000"

BATCH_SIZE = 800   # <-- diubah dari 400 -> 800 (target total foto)
DELAY_SEC  = 1.5
TIMEOUT    = 10

# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def parse_rating(raw):
    try:
        d = ast.literal_eval(str(raw))
        if isinstance(d, dict):
            return float(d.get("average", 0))
    except:
        pass
    return 0.0

def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:80]

def normalize_key(fname: str) -> str:
    """Hapus prefix angka (001_, 002_, dst) dan ekstensi → lowercase key."""
    clean = re.sub(r"^\d+_", "", fname)
    return clean.rsplit(".", 1)[0].lower()

os.makedirs(FOTO_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("Membaca dataset...")
df_raw = pd.read_csv(INPUT_CSV, low_memory=False)
df_raw["rating_avg"] = df_raw["ratings"].apply(parse_rating)
df_raw = df_raw.sort_values("rating_avg", ascending=False).reset_index(drop=True)
print(f"  Total dataset          : {len(df_raw)} kuliner")

if os.path.exists(OUTPUT_CSV):
    df_done  = pd.read_csv(OUTPUT_CSV, low_memory=False)
    uid_done = set(df_done["uid"].astype(str))
    print(f"  Sudah punya foto       : {len(uid_done)} kuliner (dari CSV)")
else:
    df_done  = pd.DataFrame()
    uid_done = set()
    print("  Output CSV belum ada, mulai dari awal")

# Scan foto di disk — normalize key (strip prefix angka)
existing_disk = {}
for fname in os.listdir(FOTO_DIR):
    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        key = normalize_key(fname)
        existing_disk[key] = os.path.join(FOTO_DIR, fname)
print(f"  Foto di disk           : {len(existing_disk)} file")

# Kuliner belum punya foto
df_missing = df_raw[~df_raw["uid"].astype(str).isin(uid_done)].copy().reset_index(drop=True)
print(f"  Belum punya foto       : {len(df_missing)} kuliner")

df_target = df_missing.head(BATCH_SIZE).copy()
df_target["photo_url"]          = ""
df_target["photo_source"]       = ""
df_target["photo_search_query"] = ""

# Hitung berapa yang sudah ada di disk (akan di-skip)
will_skip = sum(
    1 for _, r in df_target.iterrows()
    if safe_filename(str(r["displayName"])).lower() in existing_disk
)
print(f"\nTarget batch ini        : {BATCH_SIZE} kuliner")
print(f"  Sudah ada di disk      : {will_skip} → akan di-skip")
print(f"  Perlu download         : {BATCH_SIZE - will_skip}")

# ─────────────────────────────────────────────
# HTTP SESSION
# ─────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
})

# ─────────────────────────────────────────────
# FUNGSI DOWNLOAD
# ─────────────────────────────────────────────
def download_image(url: str, dest_path: str) -> bool:
    try:
        resp = SESSION.get(url, timeout=TIMEOUT, stream=True)
        ct   = resp.headers.get("Content-Type", "")
        if resp.status_code == 200 and "image" in ct:
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(1024):
                    f.write(chunk)
            if os.path.getsize(dest_path) < 2_000:
                os.remove(dest_path)
                return False
            return True
    except Exception as e:
        print(f"   Download gagal: {e}")
    return False

# ─────────────────────────────────────────────
# STRATEGI A: GoFood OG Image
# ─────────────────────────────────────────────
def try_gofood_thumb(shortlink) -> str | None:
    if not shortlink or pd.isna(shortlink):
        return None
    try:
        resp = SESSION.get(str(shortlink), timeout=TIMEOUT, allow_redirects=True)
        if resp.status_code != 200:
            return None
        for pattern in [
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        ]:
            m = re.search(pattern, resp.text)
            if m:
                return m.group(1)
    except Exception as e:
        print(f"   GoFood error: {e}")
    return None

# ─────────────────────────────────────────────
# STRATEGI B: Bing Image Search (ganti DDG)
# ─────────────────────────────────────────────
def bing_image_search(query: str, max_results: int = 10) -> list:
    """Scrape Bing Images tanpa API key."""
    try:
        params = {
            "q":    query,
            "form": "HDRSC2",
            "first": 1,
        }
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8",
            "Referer": "https://www.bing.com/",
        }
        resp = requests.get(
            "https://www.bing.com/images/search",
            params=params, headers=headers, timeout=TIMEOUT
        )
        if resp.status_code != 200:
            print(f"   Bing status: {resp.status_code}")
            return []

        # Extract image URLs dari JSON embedded di HTML
        urls = re.findall(r'"murl":"(https?://[^"]+)"', resp.text)
        # Deduplicate, filter non-image extensions
        seen  = set()
        clean = []
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            if re.search(r"\.(jpg|jpeg|png|webp)(\?|$)", u, re.I):
                clean.append(u)
            if len(clean) >= max_results:
                break
        return clean
    except Exception as e:
        print(f"   Bing error: {e}")
        return []

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("Mulai scrape foto batch missing...")
print("=" * 65)

sukses = 0
gagal  = 0
skipped_disk = 0

for i, row in df_target.iterrows():
    nama      = str(row["displayName"])
    shortlink = row.get("shortLink", "")
    rating    = row.get("rating_avg", 0)

    fname_base = safe_filename(nama)
    fname      = f"{fname_base}.jpg"
    dest_path  = os.path.join(FOTO_DIR, fname)
    foto_url   = f"{FASTAPI_BASE}/foto/{fname}"
    fname_key  = fname_base.lower()

    print(f"\n[{i+1:03d}/{BATCH_SIZE}] {nama[:55]}")
    print(f"          Rating: {rating:.2f}")

    # ✅ Skip jika file sudah ada di disk
    if fname_key in existing_disk:
        existing_path  = existing_disk[fname_key]
        existing_fname = os.path.basename(existing_path)
        print(f"   ✓ Skip — sudah ada di disk: {existing_fname}")
        df_target.at[i, "photo_local"]        = existing_path
        df_target.at[i, "photo_url"]          = f"{FASTAPI_BASE}/foto/{existing_fname}"
        df_target.at[i, "photo_source"]       = "cached_disk"
        df_target.at[i, "photo_search_query"] = ""
        skipped_disk += 1
        sukses += 1
        continue

    photo_url    = None
    photo_source = None
    used_query   = ""

    # A: GoFood OG
    og = try_gofood_thumb(shortlink)
    if og and download_image(og, dest_path):
        photo_url    = og
        photo_source = "gofood_og"
        used_query   = str(shortlink)
        print(f"   ✓ GoFood OG berhasil")

    # B: Bing Image Search fallback
    if not photo_url:
        brand   = str(row.get("brand.displayName", nama))
        queries = [
            f"{brand} Bekasi makanan",
            f"{brand} restoran Bekasi",
            f"{nama} kuliner Bekasi",
        ]
        for q in queries:
            print(f"   Bing: {q[:55]}")
            for url in bing_image_search(q, max_results=10):
                if download_image(url, dest_path):
                    photo_url    = url
                    photo_source = "bing"
                    used_query   = q
                    print(f"   ✓ Bing berhasil")
                    break
            if photo_url:
                break
            time.sleep(DELAY_SEC)

    if photo_url:
        df_target.at[i, "photo_local"]        = dest_path
        df_target.at[i, "photo_url"]          = foto_url
        df_target.at[i, "photo_source"]       = photo_source
        df_target.at[i, "photo_search_query"] = used_query
        existing_disk[fname_key] = dest_path
        sukses += 1
    else:
        print(f"   ✗ Tidak dapat foto")
        gagal += 1
        df_target.at[i, "photo_url"]    = ""
        df_target.at[i, "photo_source"] = "failed"

    time.sleep(DELAY_SEC)

# ─────────────────────────────────────────────
# SIMPAN — append ke OUTPUT_CSV
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("Menyimpan hasil...")

cols_output = [
    "uid", "ratings", "priceLevel", "key", "tenantUid", "displayName",
    "description", "countryCode", "timeZone", "status", "openPeriods",
    "createTime", "brandUid", "badges", "notes", "shortLink",
    "nextCloseTime", "tags", "location.latitude", "location.longitude",
    "serviceArea.id", "brand.key", "brand.tenantUid", "brand.uid",
    "brand.displayName", "areaName",
    "photo_url", "photo_source", "photo_search_query",
]

for col in cols_output:
    if col not in df_target.columns:
        df_target[col] = ""

df_save = df_target[cols_output].copy()

if not df_done.empty:
    for col in cols_output:
        if col not in df_done.columns:
            df_done[col] = ""
    df_done_aligned = df_done[cols_output].copy()
    new_uids        = set(df_save["uid"].astype(str))
    df_done_clean   = df_done_aligned[~df_done_aligned["uid"].astype(str).isin(new_uids)]
    df_final        = pd.concat([df_done_clean, df_save], ignore_index=True)
else:
    df_final = df_save

df_final.to_csv(OUTPUT_CSV, index=False)
print(f"  CSV disimpan ke  : {OUTPUT_CSV}")
print(f"  Total baris CSV  : {len(df_final)}")

# ─────────────────────────────────────────────
# RINGKASAN
# ─────────────────────────────────────────────
has_photo = (
    df_target["photo_url"].notna() &
    df_target["photo_url"].astype(str).str.startswith("http")
).sum()

print(f"\n{'='*65}")
print(f"SELESAI BATCH INI!")
print(f"  Target batch        : {BATCH_SIZE}")
print(f"  Sukses total        : {sukses}  (termasuk {skipped_disk} dari disk)")
print(f"  Foto baru download  : {sukses - skipped_disk}")
print(f"  Gagal               : {gagal}")
print(f"{'='*65}")

print(f"\nSOURCE BREAKDOWN (batch ini):")
print(df_target["photo_source"].value_counts(dropna=False).to_string())

print(f"\nTOP 10 HASIL BATCH INI:")
print(f"{'No':<5} {'Nama':<45} {'Rating':>7} {'Source':<12} {'Status'}")
print("-" * 78)
for rank, (_, row) in enumerate(df_target.head(10).iterrows(), 1):
    ada    = "OK" if str(row.get("photo_url","")).startswith("http") else "GAGAL"
    source = str(row.get("photo_source",""))[:11] or "-"
    print(f"{rank:<5} {str(row['displayName'])[:44]:<45} {row['rating_avg']:>7.2f} {source:<12} {ada}")

sisa = max(0, len(df_missing) - BATCH_SIZE)
print(f"\nSisa kuliner belum foto: {sisa}")
if sisa > 0:
    print("Jalankan ulang script untuk batch berikutnya.")