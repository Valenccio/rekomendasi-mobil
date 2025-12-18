import streamlit as st
import pandas as pd
import joblib
import os

from train_model_mobil import main as train_model  # pastikan file ini ada di repo

st.set_page_config(page_title="Rekomendasi Mobil Bekas", page_icon="🚗", layout="wide")

# ===========================
# CSS UI (cards + hero)
# ===========================
st.markdown("""
<style>
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1400px; }
.stApp { background: #0f1117; }

.hero {
  background: linear-gradient(135deg, rgba(12,18,28,0.92), rgba(33,42,62,0.92));
  padding: 22px 22px;
  border-radius: 18px;
  color: #fff;
  box-shadow: 0 12px 30px rgba(0,0,0,0.25);
  margin-bottom: 16px;
}
.hero-title { font-size: 2.0rem; font-weight: 850; margin: 0; }
.hero-subtitle { font-size: 1rem; opacity: 0.9; margin-top: 8px; margin-bottom: 0; }
.badges { margin-top: 12px; display: flex; gap: 10px; flex-wrap: wrap; }
.badge {
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.18);
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  backdrop-filter: blur(6px);
}

.kpi {
  background: rgba(30, 34, 39, 0.92);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.22);
  margin-bottom: 14px;
}

.card {
  background: rgba(30, 34, 39, 0.92);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.22);
  margin-bottom: 14px;
}

.car-title { font-size: 1.05rem; font-weight: 800; margin: 0 0 6px 0; }
.car-sub { color: rgba(255,255,255,0.75); margin: 0 0 10px 0; font-size: 0.92rem; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.chip {
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.12);
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.83rem;
}

.big-price { font-size: 1.55rem; font-weight: 900; margin: 6px 0 2px 0; }
.small { color: rgba(255,255,255,0.75); font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ===========================
# Load data & model
# ===========================
@st.cache_data
def load_data():
    return pd.read_csv("data_mobil_bekas.csv")

@st.cache_resource
def load_models():
    if (not os.path.exists("model_harga_mobil.pkl")) or (not os.path.exists("model_score_mobil.pkl")):
        with st.status("Model belum tersedia. Sedang melatih model…", expanded=True) as status:
            st.write("Menyiapkan model prediksi harga…")
            st.write("Menyiapkan model skor kecocokan…")
            train_model()
            status.update(label="Model selesai dilatih ✅", state="complete")

    price_model = joblib.load("model_harga_mobil.pkl")
    score_model = joblib.load("model_score_mobil.pkl")
    return price_model, score_model

df = load_data()
price_model, score_model = load_models()

FEATURE_COLS = [
    "merk", "model", "segmen", "tahun", "kilometer", "transmisi", "bahan_bakar",
    "kapasitas_cc", "warna", "kota", "jumlah_pemilik", "riwayat_servis",
    "bekas_banjir", "bekas_tabrak", "pajak_hidup", "budget_user_juta"
]

# Rapikan numerik
num_cols = ["tahun","kilometer","kapasitas_cc","jumlah_pemilik","budget_user_juta","harga_jual_juta","rekomendasi_score"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    st.error(f"Dataset kurang kolom: {missing}")
    st.stop()

df = df.dropna(subset=FEATURE_COLS).copy()

# ===========================
# Header / Hero
# ===========================
st.markdown(f"""
<div class="hero">
  <div class="hero-title">🚗 Rekomendasi Mobil Bekas</div>
  <div class="hero-subtitle">Masukkan budget & preferensi. Sistem akan memprediksi harga dan meranking mobil paling cocok.</div>
  <div class="badges">
    <span class="badge">✅ Prediksi Harga (ML)</span>
    <span class="badge">✅ Skor Kecocokan</span>
    <span class="badge">✅ Filter Risiko</span>
    <span class="badge">✅ Top-N Rekomendasi</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ===========================
# Sidebar filters (dirapikan)
# ===========================
st.sidebar.header("Preferensi")

default_budget = float(df["budget_user_juta"].median()) if df["budget_user_juta"].notna().any() else 200.0
budget = st.sidebar.number_input("Budget (juta rupiah)", min_value=10.0, max_value=2000.0, value=default_budget, step=5.0)
top_n = st.sidebar.slider("Jumlah rekomendasi", 3, 20, 8)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Utama")

def opts(col):
    return ["Semua"] + sorted(df[col].dropna().unique().tolist())

pil_merk = st.sidebar.selectbox("Merk", opts("merk"), index=0)
pil_segmen = st.sidebar.selectbox("Segmen", opts("segmen"), index=0)
pil_trans = st.sidebar.selectbox("Transmisi", opts("transmisi"), index=0)
pil_bbm = st.sidebar.selectbox("Bahan bakar", opts("bahan_bakar"), index=0)
pil_kota = st.sidebar.selectbox("Kota", opts("kota"), index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Range")

min_tahun, max_tahun = int(df["tahun"].min()), int(df["tahun"].max())
tahun_range = st.sidebar.slider("Tahun", min_tahun, max_tahun, (min_tahun, max_tahun), step=1)

km_min, km_max = int(df["kilometer"].min()), int(df["kilometer"].max())
km_range = st.sidebar.slider("Kilometer", km_min, km_max, (km_min, km_max), step=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("Risiko & Pajak")

tanpa_banjir = st.sidebar.checkbox("Bukan bekas banjir", value=True)
tanpa_tabrak = st.sidebar.checkbox("Bukan bekas tabrak", value=True)
pajak_hidup_only = st.sidebar.checkbox("Pajak hidup saja", value=False)

# helper boolean text
def is_no(x):
    return str(x).strip().lower() in ["tidak","no","false","0"]

def is_yes(x):
    return str(x).strip().lower() in ["ya","yes","true","1"]

# ===========================
# Filter data
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

if "bekas_banjir" in df_f.columns and tanpa_banjir:
    df_f = df_f[df_f["bekas_banjir"].apply(is_no)]
if "bekas_tabrak" in df_f.columns and tanpa_tabrak:
    df_f = df_f[df_f["bekas_tabrak"].apply(is_no)]
if "pajak_hidup" in df_f.columns and pajak_hidup_only:
    df_f = df_f[df_f["pajak_hidup"].apply(is_yes)]

# ===========================
# Predict + rank
# ===========================
tab1, tab2, tab3 = st.tabs(["⭐ Rekomendasi", "📄 Tabel Lengkap", "ℹ️ Cara Penilaian"])

with tab1:
    if df_f.empty:
        st.warning("Tidak ada listing yang cocok dengan filter. Coba longgarkan filter.")
        st.stop()

    X_pred = df_f[FEATURE_COLS].copy()
    X_pred["budget_user_juta"] = budget

    df_f = df_f.copy()
    df_f["pred_harga_juta"] = price_model.predict(X_pred)
    df_f["pred_score"] = score_model.predict(X_pred)

    df_budget = df_f[df_f["pred_harga_juta"] <= budget].copy()
    if df_budget.empty:
        st.warning("Setelah diprediksi, tidak ada yang masuk budget. Coba naikkan budget.")
        st.stop()

    df_budget = df_budget.sort_values("pred_score", ascending=False).head(top_n).copy()

    # KPI row
    best = df_budget.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.metric("Listing cocok", f"{len(df_budget)}")
        st.markdown('</div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.metric("Min harga prediksi", f"{df_budget['pred_harga_juta'].min():.1f} jt")
        st.markdown('</div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.metric("Max harga prediksi", f"{df_budget['pred_harga_juta'].max():.1f} jt")
        st.markdown('</div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.metric("Skor rata-rata", f"{df_budget['pred_score'].mean():.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"✅ **Rekomendasi terbaik:** **{best['merk']} {best['model']} ({int(best['tahun'])})** — "
                f"prediksi **{best['pred_harga_juta']:.1f} jt**, skor **{best['pred_score']:.1f}**")

    # Cards for Top results
    st.markdown("### Top Rekomendasi")
    cols = st.columns(2)
    for i, row in enumerate(df_budget.itertuples(index=False)):
        col = cols[i % 2]
        with col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<div class='car-title'>{row.merk} {row.model} ({int(row.tahun)})</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='car-sub'>{row.segmen} • {row.kota} • Warna: {row.warna}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-price'>{row.pred_harga_juta:.1f} jt</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small'>Skor kecocokan: <b>{row.pred_score:.1f}</b></div>", unsafe_allow_html=True)

            km_txt = f"{int(row.kilometer):,} km".replace(",", ".")
            cc_txt = f"{int(row.kapasitas_cc)} cc"
            chips = [
                f"Transmisi: {row.transmisi}",
                f"BBM: {row.bahan_bakar}",
                f"KM: {km_txt}",
                f"CC: {cc_txt}",
                f"Pemilik: {int(row.jumlah_pemilik)}",
                f"Servis: {row.riwayat_servis}",
                f"Banjir: {row.bekas_banjir}",
                f"Tabrak: {row.bekas_tabrak}",
                f"Pajak: {row.pajak_hidup}",
            ]
            st.markdown("<div class='chips'>" + "".join([f"<span class='chip'>{c}</span>" for c in chips]) + "</div>", unsafe_allow_html=True)

            # detail (opsional)
            with st.expander("Detail"):
                # tampilkan kolom penting saja
                detail_cols = ["id_listing","merk","model","segmen","tahun","kilometer","transmisi","bahan_bakar",
                               "kapasitas_cc","warna","kota","jumlah_pemilik","riwayat_servis","bekas_banjir",
                               "bekas_tabrak","pajak_hidup","pred_harga_juta","pred_score"]
                detail_cols = [c for c in detail_cols if c in df_budget.columns]
                st.write(df_budget.iloc[i][detail_cols].to_frame("value"))

            st.markdown("</div>", unsafe_allow_html=True)

    # Download CSV
    csv = df_budget.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download hasil rekomendasi (CSV)", csv, "hasil_rekomendasi.csv", "text/csv")

with tab2:
    st.markdown("### Tabel Lengkap (setelah filter & prediksi)")
    st.caption("Tip: Jika tabel terlalu lebar, lihat yang penting di tab Rekomendasi (cards).")

    if df_f.empty:
        st.info("Belum ada data (filter terlalu ketat).")
    else:
        # prediksi dulu supaya tabel lengkap punya kolom pred_*
        X_pred = df_f[FEATURE_COLS].copy()
        X_pred["budget_user_juta"] = budget
        df_tmp = df_f.copy()
        df_tmp["pred_harga_juta"] = price_model.predict(X_pred)
        df_tmp["pred_score"] = score_model.predict(X_pred)

        # tampilkan kolom lebih relevan
        cols_show = [
            "id_listing","merk","model","segmen","tahun","kilometer","transmisi","bahan_bakar",
            "kapasitas_cc","warna","kota","jumlah_pemilik","riwayat_servis","bekas_banjir","bekas_tabrak","pajak_hidup",
            "pred_harga_juta","pred_score"
        ]
        cols_show = [c for c in cols_show if c in df_tmp.columns]
        st.dataframe(df_tmp[cols_show].reset_index(drop=True), use_container_width=True)

with tab3:
    st.markdown("### Cara Penilaian (singkat)")
    st.write("""
- Sistem melakukan **prediksi harga** (juta rupiah) dan **prediksi skor kecocokan**.
- Listing yang ditampilkan adalah yang **masuk budget** (pred_harga_juta ≤ budget).
- Setelah itu, mobil diurutkan berdasarkan **pred_score tertinggi**.
""")
    st.info("Catatan: skor & harga bergantung pada kualitas dataset. Semakin banyak data, semakin stabil hasilnya.")
