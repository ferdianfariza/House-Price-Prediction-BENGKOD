import json
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import FuncFormatter

# ---------- Konfigurasi ----------
BASE = Path(__file__).parent

st.set_page_config(
    page_title="Estimasi Harga Rumah",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

mpl.rcParams["axes.formatter.use_mathtext"] = False
plt.rcParams["savefig.transparent"] = True

# ---------- Design System (tanpa warna hardcode) ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp, button, input, textarea, label {
        font-family: 'Inter', sans-serif !important;
    }

    .block-container { max-width: 1180px; padding: 2.5rem 2rem 4rem; }

    h1 {
        font-size: 1.85rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.04em;
        line-height: 1.25 !important;
    }
    h2, h3 { font-weight: 600 !important; letter-spacing: -0.03em; }

    .eyebrow {
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
        color: var(--primary-color); text-transform: uppercase; margin-bottom: 0.5rem;
    }
    .section-title {
        font-size: 1.05rem; font-weight: 600; letter-spacing: -0.02em; margin-bottom: 0.3rem;
    }
    .section-description {
        opacity: .65; font-size: 0.85rem; line-height: 1.6; margin-bottom: 1.2rem;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1.5rem; }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        height: 44px; padding: 0 0.1rem; font-size: 0.9rem; font-weight: 500;
    }

    [data-testid="stMetric"] { padding: 0.35rem 0; background: transparent; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 600; letter-spacing: -0.04em; }

    .price-label {
        opacity: .65; font-size: 0.78rem; font-weight: 500; margin-bottom: 0.4rem;
    }
    .price-value {
        font-size: clamp(1.8rem, 3vw, 2.5rem); line-height: 1.2;
        font-weight: 600; letter-spacing: -0.05em; overflow-wrap: anywhere;
    }
    .price-unit { opacity: .65; font-size: 0.82rem; margin-top: 0.55rem; }

    .range-row {
        display: flex; justify-content: space-between; gap: 1rem;
        padding: 0.65rem 0; border-bottom: 1px solid rgba(128,128,128,.25); font-size: 0.83rem;
    }
    .range-label { opacity: .65; }
    .range-value { font-weight: 600; text-align: right; }

    .result-note {
        background: rgba(128,128,128,.10); border: 1px solid rgba(128,128,128,.25);
        border-radius: 8px; padding: 0.85rem 1rem; font-size: 0.82rem; line-height: 1.6;
    }

    .step { display: flex; gap: 1rem; padding: 1rem 0; border-bottom: 1px solid rgba(128,128,128,.25); }
    .step .no { color: var(--primary-color); font-size: 0.8rem; font-weight: 600; padding-top: .15rem; }
    .step .t { font-size: 0.9rem; font-weight: 600; }
    .step .d { font-size: 0.8rem; opacity: .65; margin-top: .25rem; }

    .stButton > button, .stFormSubmitButton > button {
        min-height: 44px; border-radius: 8px; font-weight: 500;
    }
    [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] > div > div { border-radius: 8px; }
    [data-testid="stAlert"] { border-radius: 8px; }
    [data-testid="stDataFrame"], div[data-testid="stExpander"] { border-radius: 8px; }

    footer { visibility: hidden; }

    @media (max-width: 768px) {
        .block-container { padding: 1.5rem 1rem 3rem; }
        h1 { font-size: 1.55rem !important; }
        [data-testid="stMetricValue"] { font-size: 1.25rem; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 1rem; }
        [data-testid="stTabs"] [data-baseweb="tab"] { font-size: 0.82rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Load Resource ----------
@st.cache_resource
def muat_model():
    return joblib.load(BASE / "model" / "model_rf.joblib")


@st.cache_resource
def muat_kolom():
    return joblib.load(BASE / "model" / "kolom.joblib")


@st.cache_data
def muat_data():
    return pd.read_csv(BASE / "data_bersih.csv")


@st.cache_data
def muat_meta():
    with open(BASE / "model" / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def rupiah(juta):
    """Format harga dalam juta rupiah."""
    juta = float(juta)
    if not np.isfinite(juta):
        return "Tidak tersedia"
    if juta < 0:
        return "-Rp " + f"{abs(juta):,.0f}".replace(",", ".") + " Juta"
    if juta >= 1000:
        teks = (
            f"{juta / 1000:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
        return f"Rp {teks} Miliar"
    return "Rp " + f"{juta:,.0f}".replace(",", ".") + " Juta"


# ---------- Warna grafik mengikuti tema aktif ----------
def warna_tema():
    try:
        gelap = st.context.theme.type == "dark"
        teks = "#FAFAFA" if gelap else "#31333F"
    except Exception:
        teks = "#888888"  # netral, terbaca di light maupun dark
    aksen = st.get_option("theme.primaryColor") or "#FF4B4B"
    return teks, aksen


TEKS, AKSEN = warna_tema()


def fmt_angka(v, _):
    return f"{v:,.0f}".replace(",", ".") if v >= 1 else f"{v:g}"


def rapikan_grafik(ax, fig, grid_axis="y"):
    """Gaya grafik minimalis, transparan, dan tanpa mathtext."""
    for sumbu in (ax.xaxis, ax.yaxis):
        if sumbu.get_scale() == "log":
            sumbu.set_major_formatter(FuncFormatter(fmt_angka))
            sumbu.set_minor_formatter(FuncFormatter(lambda v, _: ""))

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(TEKS)
        ax.spines[s].set_alpha(0.3)

    ax.tick_params(axis="both", colors=TEKS, labelsize=8, length=0, pad=6)
    ax.xaxis.label.set_color(TEKS)
    ax.yaxis.label.set_color(TEKS)
    ax.grid(axis=grid_axis, which="major", color=TEKS, alpha=0.12, linewidth=0.7)
    ax.set_axisbelow(True)

    leg = ax.get_legend()
    if leg:
        for t in leg.get_texts():
            t.set_color(TEKS)

    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    fig.tight_layout()


# ---------- Inisialisasi ----------
try:
    model = muat_model()
    kolom = muat_kolom()
    df = muat_data()
    meta = muat_meta()
except (FileNotFoundError, OSError, ValueError) as e:
    st.error(f"Gagal memuat resource aplikasi: {e}")
    st.stop()

if df.empty:
    st.error("Dataset kosong. Periksa file data_bersih.csv.")
    st.stop()

M = meta["metrik_test"]
MAPE = float(M["MAPE_%"])

KECAMATAN = sorted(
    c.replace("kecamatan_", "", 1) for c in kolom if c.startswith("kecamatan_")
)
FITUR_NUM = ["kamar_tidur", "kamar_mandi", "luas_tanah", "luas_bangunan"]
rentang = {c: (float(df[c].min()), float(df[c].max())) for c in FITUR_NUM}

# ---------- Header ----------
st.markdown('<div class="eyebrow">Property analytics</div>', unsafe_allow_html=True)
st.title("Estimasi harga rumah")
st.caption("Depok, Jawa Barat · Prediksi berbasis data listing")
st.write("")

tab_pred, tab_eda, tab_model = st.tabs(["Prediksi", "Eksplorasi data", "Tentang model"])


# ==========================================================
# TAB 1 — PREDIKSI
# ==========================================================
with tab_pred:
    st.write("")
    kiri, kanan = st.columns([0.9, 1.1], gap="large")

    with kiri:
        st.markdown(
            '<div class="section-title">Spesifikasi properti</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-description">Masukkan karakteristik rumah untuk mendapatkan estimasi harga.</div>',
            unsafe_allow_html=True,
        )

        with st.form("form_prediksi"):
            kecamatan = st.selectbox(
                "Lokasi",
                KECAMATAN,
                index=(KECAMATAN.index("Cinere") if "Cinere" in KECAMATAN else 0),
            )
            c1, c2 = st.columns(2)
            kt = c1.number_input(
                "Kamar tidur", min_value=1, max_value=15, value=3, step=1
            )
            km = c2.number_input(
                "Kamar mandi", min_value=1, max_value=15, value=2, step=1
            )
            c3, c4 = st.columns(2)
            lt = c3.number_input(
                "Luas tanah (m²)", min_value=10, max_value=5000, value=90, step=5
            )
            lb = c4.number_input(
                "Luas bangunan (m²)", min_value=10, max_value=3000, value=70, step=5
            )
            st.write("")
            kirim = st.form_submit_button(
                "Hitung estimasi", type="primary", width="stretch"
            )

    with kanan:
        if kirim:
            x = pd.DataFrame(
                {
                    "kamar_tidur": [kt],
                    "kamar_mandi": [km],
                    "luas_tanah": [lt],
                    "luas_bangunan": [lb],
                    "kecamatan": [kecamatan],
                }
            )
            x = pd.get_dummies(x, columns=["kecamatan"], dtype=float).reindex(
                columns=kolom, fill_value=0
            )

            pred = float(model.predict(x)[0]) / 1e6  # Rupiah -> Juta
            mape = MAPE / 100

            st.markdown(
                '<div class="price-label">ESTIMASI HARGA PENAWARAN</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="price-value">{rupiah(pred)}</div>', unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="price-unit">Kisaran indikatif berdasarkan MAPE {MAPE:.1f}%</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(
                f"""
                <div class="range-row"><span class="range-label">Estimasi bawah</span>
                    <span class="range-value">{rupiah(pred * (1 - mape))}</span></div>
                <div class="range-row"><span class="range-label">Estimasi atas</span>
                    <span class="range-value">{rupiah(pred * (1 + mape))}</span></div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.markdown(
                '<div class="section-title">Ringkasan properti</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"{kecamatan} · {kt} kamar tidur · {km} kamar mandi")
            st.caption(
                f"Luas tanah {lt:,} m² · Luas bangunan {lb:,} m²".replace(",", ".")
            )

            peringatan = []
            if lb > 6 * lt:
                peringatan.append(
                    "Luas bangunan lebih dari enam kali luas tanah. Proporsi ini tidak lazim."
                )
            for nama, v in [
                ("kamar_tidur", kt),
                ("kamar_mandi", km),
                ("luas_tanah", lt),
                ("luas_bangunan", lb),
            ]:
                lo, hi = rentang[nama]
                if not lo <= v <= hi:
                    peringatan.append(
                        f"{nama.replace('_', ' ').title()} berada di luar rentang data latih ({lo:.0f}–{hi:.0f})."
                    )
            for pesan in peringatan:
                st.warning(pesan)

            st.write("")
            st.markdown(
                '<div class="result-note">Estimasi ini merupakan prediksi model, bukan '
                "penilaian resmi atau jaminan harga transaksi.</div>",
                unsafe_allow_html=True,
            )

            st.write("")
            st.markdown(
                '<div class="section-title">Posisi terhadap listing</div>',
                unsafe_allow_html=True,
            )
            st.caption("Harga dan luas bangunan ditampilkan pada skala logaritmik.")

            sebaran = df[df["kecamatan"] == kecamatan]
            pakai_lokasi = len(sebaran) >= 50

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.scatter(
                df["luas_bangunan"].to_numpy(),
                df["harga_juta"].to_numpy(),
                s=7,
                alpha=0.15,
                color=TEKS,
                linewidths=0,
                label="Semua listing",
            )
            if pakai_lokasi:
                ax.scatter(
                    sebaran["luas_bangunan"].to_numpy(),
                    sebaran["harga_juta"].to_numpy(),
                    s=9,
                    alpha=0.45,
                    color=AKSEN,
                    linewidths=0,
                    label=kecamatan,
                )
            ax.scatter(
                [lb],
                [pred],
                s=150,
                marker="*",
                color=AKSEN,
                edgecolors=TEKS,
                linewidths=0.8,
                zorder=5,
                label="Prediksi",
            )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Luas bangunan (m²)", fontsize=9)
            ax.set_ylabel("Harga (juta rupiah)", fontsize=9)
            ax.legend(frameon=False, fontsize=8, loc="best")

            rapikan_grafik(ax, fig, grid_axis="both")
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        else:
            st.markdown(
                '<div class="section-title">Estimasi harga rumah</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-description">Hasil prediksi akan ditampilkan di sini setelah '
                "spesifikasi properti dihitung.</div>",
                unsafe_allow_html=True,
            )
            st.divider()
            st.markdown(
                '<div class="price-label">YANG PERLU DISIAPKAN</div>',
                unsafe_allow_html=True,
            )

            for nomor, judul, detail in [
                ("01", "Lokasi", "Kecamatan atau kawasan rumah"),
                ("02", "Spesifikasi", "Jumlah kamar tidur dan kamar mandi"),
                ("03", "Luas properti", "Luas tanah dan luas bangunan"),
            ]:
                st.markdown(
                    f'<div class="step"><div class="no">{nomor}</div>'
                    f'<div><div class="t">{judul}</div><div class="d">{detail}</div></div></div>',
                    unsafe_allow_html=True,
                )


# ==========================================================
# TAB 2 — EKSPLORASI DATA
# ==========================================================
with tab_eda:
    st.write("")
    st.markdown(
        '<div class="section-title">Gambaran dataset</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-description">Distribusi harga, karakteristik properti, dan perbedaan harga antar lokasi.</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total listing", f"{len(df):,}".replace(",", "."))
    k2.metric("Median harga", rupiah(df["harga_juta"].median()))
    k3.metric(
        "Median luas tanah", f"{df['luas_tanah'].median():,.0f} m²".replace(",", ".")
    )
    k4.metric(
        "Median luas bangunan",
        f"{df['luas_bangunan'].median():,.0f} m²".replace(",", "."),
    )

    st.divider()

    a, b = st.columns(2, gap="large")

    with a:
        st.markdown(
            '<div class="section-title">Distribusi harga</div>', unsafe_allow_html=True
        )
        st.caption("Jumlah listing berdasarkan log10 harga.")
        harga_positif = df.loc[df["harga_juta"] > 0, "harga_juta"].to_numpy()

        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.hist(
            np.log10(harga_positif), bins=40, color=AKSEN, alpha=0.85, edgecolor="none"
        )
        ax.set_xlabel("log10(harga dalam juta rupiah)", fontsize=9)
        ax.set_ylabel("Jumlah listing", fontsize=9)
        rapikan_grafik(ax, fig)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with b:
        st.markdown(
            '<div class="section-title">Harga per lokasi</div>', unsafe_allow_html=True
        )
        st.caption("Median harga untuk lokasi dengan minimal 100 listing.")
        med = (
            df.groupby("kecamatan")["harga_juta"]
            .agg(median="median", n="size")
            .reset_index()
            .query("n >= 100")
            .sort_values("median")
        )

        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.barh(
            med["kecamatan"].tolist(),
            med["median"].to_numpy(),
            color=AKSEN,
            height=0.68,
        )
        ax.set_xlabel("Median harga (juta rupiah)", fontsize=9)
        ax.set_ylabel("")
        rapikan_grafik(ax, fig, grid_axis="x")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    a, b = st.columns(2, gap="large")

    with a:
        st.markdown(
            '<div class="section-title">Luas tanah dan harga</div>',
            unsafe_allow_html=True,
        )
        st.caption("Hubungan luas tanah dengan harga listing.")

        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.scatter(
            df["luas_tanah"].to_numpy(),
            df["harga_juta"].to_numpy(),
            s=8,
            alpha=0.22,
            color=AKSEN,
            linewidths=0,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Luas tanah (m²)", fontsize=9)
        ax.set_ylabel("Harga (juta rupiah)", fontsize=9)
        rapikan_grafik(ax, fig, grid_axis="both")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with b:
        st.markdown(
            '<div class="section-title">Kamar tidur dan harga</div>',
            unsafe_allow_html=True,
        )
        st.caption("Median harga berdasarkan jumlah kamar tidur.")
        m_kt = (
            df[df["kamar_tidur"] <= 8]
            .groupby("kamar_tidur", as_index=False)["harga_juta"]
            .median()
            .sort_values("kamar_tidur")
        )

        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(
            m_kt["kamar_tidur"].to_numpy(),
            m_kt["harga_juta"].to_numpy(),
            color=AKSEN,
            linewidth=2,
            marker="o",
            markersize=5,
        )
        ax.set_xlabel("Jumlah kamar tidur", fontsize=9)
        ax.set_ylabel("Median harga (juta rupiah)", fontsize=9)
        ax.set_xticks(m_kt["kamar_tidur"].to_numpy())
        rapikan_grafik(ax, fig)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.write("")
    with st.expander("Lihat data listing"):
        st.caption(
            f"Menampilkan maksimal 100 dari {len(df):,} listing.".replace(",", ".")
        )
        st.dataframe(df.head(100), width="stretch", hide_index=True)


# ==========================================================
# TAB 3 — TENTANG MODEL
# ==========================================================
with tab_model:
    st.write("")
    st.markdown(
        '<div class="section-title">Performa model</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-description">Metrik evaluasi model pada test set.</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", rupiah(M["MAE"] / 1e6))
    m2.metric("MAPE", f"{MAPE:.1f}%")
    m3.metric("R²", f"{M['R2']:.3f}")
    m4.metric("RMSE", rupiah(M["RMSE"] / 1e6))

    st.caption("Evaluasi menggunakan test set bersih sebesar 20% data.")
    st.divider()

    st.markdown(
        '<div class="section-title">Informasi model</div>', unsafe_allow_html=True
    )
    st.write("")

    info_kiri, info_kanan = st.columns(2, gap="large")

    with info_kiri:
        st.markdown('<div class="price-label">ALGORITMA</div>', unsafe_allow_html=True)
        st.markdown(f"**{meta['model']}**")
        st.markdown(
            '<div class="price-label">PREPROCESSING</div>', unsafe_allow_html=True
        )
        st.markdown("One-hot encoding untuk lokasi.")
        st.markdown('<div class="price-label">TARGET</div>', unsafe_allow_html=True)
        st.markdown("`log1p(harga)`")
        st.markdown(
            '<div class="price-label">SUMBER DATA</div>', unsafe_allow_html=True
        )
        st.markdown("Listing rumah di Depok dan sekitarnya.")

    with info_kanan:
        st.markdown(
            '<div class="price-label">FITUR INPUT</div>', unsafe_allow_html=True
        )
        for nama, deskripsi in [
            ("Lokasi", "Kecamatan atau kawasan"),
            ("Kamar tidur", "Jumlah kamar tidur"),
            ("Kamar mandi", "Jumlah kamar mandi"),
            ("Luas tanah", "Meter persegi"),
            ("Luas bangunan", "Meter persegi"),
        ]:
            st.markdown(
                f'<div class="range-row"><span class="range-label">{nama}</span>'
                f'<span class="range-value">{deskripsi}</span></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.divider()

    st.markdown(
        '<div class="section-title">Cara membaca hasil</div>', unsafe_allow_html=True
    )
    st.markdown(
        """
        - **MAE** menunjukkan rata-rata besar kesalahan absolut prediksi.
        - **MAPE** menunjukkan rata-rata kesalahan persentase absolut.
        - **R²** menunjukkan seberapa baik model menjelaskan variasi target pada data evaluasi.
        - **RMSE** memberi penalti lebih besar pada kesalahan prediksi yang besar dibandingkan MAE.
        """
    )

    st.write("")
    st.markdown(
        '<div class="result-note"><strong>Batasan model.</strong> Harga yang diprediksi adalah '
        "harga penawaran pada listing, bukan harga transaksi aktual. Kisaran berbasis MAPE "
        "merupakan gambaran kasar, bukan interval prediksi dengan jaminan cakupan statistik. "
        "Akurasi dapat berubah untuk properti yang berbeda dari distribusi data latih.</div>",
        unsafe_allow_html=True,
    )

# ---------- Footer ----------
st.write("")
st.divider()
st.caption("Estimasi Harga Rumah · Depok, Jawa Barat")
