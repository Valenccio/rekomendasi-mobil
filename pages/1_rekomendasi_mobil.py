import streamlit as st
import pandas as pd
import joblib
import os

# Pastikan file ini ada di repo dan punya fungsi main()
from train_model_mobil import main as train_model

st.set_page_config(page_title="Rekomendasi Mobil Bekas", page_icon="🚗", layout="wide")

# ===========================
# Load data & model
# ===========================
@st.cache_data
def load_data():
    return pd.read_csv("data_mobil_bekas.csv")

@st.cache_resource
def load_models():
    # Train jika model belum ada
    if (not os.path.exists("model_harga_mobil.pkl")) or (not os.path.exists("model_score_mobil.pkl")):
        st.info("Model belum tersedia. Sedang melatih model, mohon tunggu...")
        train_model()

    price_model = joblib.load("model_harga_mobil.pkl")
    score_model = joblib.load("model_score_mobil.pkl")
    return price_model, score_model

df = load_data()
price_model, score_model = load_models()

# ===========================
# Kolom fitur dataset kamu
# ===========================
FEATURE_COLS = [
    "merk", "model", "segmen", "tahun", "kilometer", "transmisi", "bahan_bakar",
    "kapasitas_cc", "warna", "kota", "jumlah_pemilik", "riwayat_servis",
    "bekas_banjir", "bekas_tabrak", "pajak_hidup", "budget_user_juta"
]

# ===========================
# Rapikan numeric
# ===========================
for c in ["tahun", "kilometer", "kapasitas_cc", "jumlah_pemilik", "budget_user_juta", "harga_jual_juta", "rekomendasi_score"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Pastikan kolom wajib ada
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    st.error(f"Dataset kamu kurang kolom: {missing}")
    st.stop()

df = df.dropna(subset=FEATURE_COLS).copy()

# ===========================
# UI
# ===========================
st.title("🚗 Rekomendasi Mobil Bekas")
st.write("Sistem akan memprediksi **harga** dan **skor kecocokan** berdasarkan input budget & preferensi kamu.")

st.sidebar.header("Preferensi")

# fallback jika kolom budget_user_juta kosong
default_budget = float(df["budget_user_juta"].median()) if df["budget_user_juta"].notna().any() else 200.0

budget = st.sidebar.number_input(
    "Budget (juta rupiah)",
    min_value=10.0,
    max_value=2000.0,
    value=default_budget,
    step=5.0
)

top_n = st.sidebar.slider("Jumlah rekomendasi", 3, 20, 8)

merk_opts = ["Semua"] + sorted(df["merk"].dropna().unique().tolist())
segmen_opts = ["Semua"] + sorted(df["segmen"].dropna().unique().tolist())
trans_opts = ["Semua"] + sorted(df["transmisi"].dropna().unique().tolist())
bbm_opts = ["Semua"] + sorted(df["bahan_bakar"].dropna().unique().tolist())
kota_opts = ["Semua"] + sorted(df["kota"].dropna().unique().tolist())

pil_merk = st.sidebar.selectbox("Merk", merk_opts, index=0)
pil_segmen = st.sidebar.selectbox("Segmen", segmen_opts, index=0)
pil_trans = st.sidebar.selectbox("Transmisi", trans_opts, index=0)
pil_bbm = st.sidebar.selectbox("Bahan bakar", bbm_opts, index=0)
pil_kota = st.sidebar.selectbox("Kota", kota_opts, index=0)

min_tahun, max_tahun = int(df["tahun"].min()), int(df["tahun"].max())
tahun_range = st.sidebar.slider("Rentang tahun", min_tahun, max_tahun, (min_tahun, max_tahun), step=1)

km_min, km_max = int(df["kilometer"].min()), int(df["kilometer"].max())
km_range = st.sidebar.slider("Rentang kilometer", km_min, km_max, (km_min, km_max), step=1000)

tanpa_banjir = st.sidebar.checkbox("Hanya yang bukan bekas banjir", value=True)
tanpa_tabrak = st.sidebar.checkbox("Hanya yang bukan bekas tabrak", value=True)
pajak_hidup_only = st.sidebar.checkbox("Hanya yang pajak hidup", value=False)

# ===========================
# Filter data statis
# ===========================
df_f = df.copy()

if pil_merk != "Semua":
    df_f = df_f[df_f["merk"] == pil_merk]
if pil_segmen != "Semua":
    df_f = df_f[df_f["segmen"] == pil_segmen]
if pil_trans != "Semua":
    df_f = df_f[df_f["transmisi"] == pil_trans]
if pil_bbm != "Semua":
    df_f = df_f[df_f["bahan_bakar"] == pil_bbm]
if pil_kota != "Semua":
    df_f = df_f[df_f["kota"] == pil_kota]

df_f = df_f[df_f["tahun"].between(tahun_range[0], tahun_range[1])]
df_f = df_f[df_f["kilometer"].between(km_range[0], km_range[1])]

# Filter boolean/text yang aman (Ya/Tidak/True/False)
def is_no(x):
    return str(x).strip().lower() in ["tidak", "no", "false", "0"]

def is_yes(x):
    return str(x).strip().lower() in ["ya", "yes", "true", "1"]

if "bekas_banjir" in df_f.columns and tanpa_banjir:
    df_f = df_f[df_f["bekas_banjir"].apply(is_no)]

if "bekas_tabrak" in df_f.columns and tanpa_tabrak:
    df_f = df_f[df_f["bekas_tabrak"].apply(is_no)]

if "pajak_hidup" in df_f.columns and pajak_hidup_only:
    df_f = df_f[df_f["pajak_hidup"].apply(is_yes)]

st.subheader("Hasil Rekomendasi")

if df_f.empty:
    st.warning("Tidak ada listing yang cocok dengan filter. Coba longgarkan filter.")
    st.stop()

# ===========================
# Prediksi (override budget)
# ===========================
X_pred = df_f[FEATURE_COLS].copy()
X_pred["budget_user_juta"] = budget

df_f = df_f.copy()
df_f["pred_harga_juta"] = price_model.predict(X_pred)
df_f["pred_score"] = score_model.predict(X_pred)

# Masuk budget berdasarkan prediksi harga
df_f = df_f[df_f["pred_harga_juta"] <= budget].copy()
if df_f.empty:
    st.warning("Setelah diprediksi, tidak ada yang masuk budget. Coba naikkan budget.")
    st.stop()

df_f = df_f.sort_values("pred_score", ascending=False).head(top_n)

def juta(x):
    return f"{x:,.1f} jt".replace(",", ".")

cols_show = [
    "id_listing","merk","model","segmen","tahun","kilometer","transmisi","bahan_bakar",
    "kapasitas_cc","warna","kota","jumlah_pemilik","riwayat_servis","bekas_banjir","bekas_tabrak","pajak_hidup"
]
cols_show = [c for c in cols_show if c in df_f.columns]

tampil = df_f[cols_show].copy()
tampil["Pred Harga"] = df_f["pred_harga_juta"].apply(juta)
tampil["Pred Score"] = df_f["pred_score"].round(1)

if "kilometer" in tampil.columns:
    tampil["kilometer"] = tampil["kilometer"].astype(int).map(lambda v: f"{v:,} km".replace(",", "."))
if "kapasitas_cc" in tampil.columns:
    tampil["kapasitas_cc"] = tampil["kapasitas_cc"].astype(int)

st.dataframe(tampil.reset_index(drop=True), use_container_width=True)

best = df_f.iloc[0]
st.success(
    f"✅ Rekomendasi terbaik: **{best['merk']} {best['model']} ({int(best['tahun'])})** "
    f"— prediksi harga **{juta(best['pred_harga_juta'])}**, skor **{best['pred_score']:.1f}**"
)

with st.expander("Detail perhitungan"):
    st.write("Budget input (juta):", budget)
    st.write("Model: prediksi harga + prediksi skor, lalu ranking berdasarkan skor.")
