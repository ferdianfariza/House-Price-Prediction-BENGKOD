import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
st.set_page_config(
    page_title="Prediksi Harga Rumah Depok", page_icon="🏠", layout="wide"
)


# ---------- Load resource ----------
@st.cache_resource
def muat_model():
    return joblib.load(BASE / "model_harga_rumah.joblib")


@st.cache_data
def muat_data():
    return pd.read_csv(BASE / "data_bersih.csv")


@st.cache_data
def muat_meta():
    return json.load(open(BASE / "meta_model.json"))


def rupiah(juta):
    """Format harga (Juta Rupiah) ke teks Indonesia."""
    if juta >= 1000:
        teks = (
            f"{juta / 1000:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
        return f"Rp {teks} Miliar"
    return "Rp " + f"{juta:,.0f}".replace(",", ".") + " Juta"


model, df, meta = muat_model(), muat_data(), muat_meta()
rentang, M = meta["rentang"], meta["metrik_test"]

st.title("Prediksi Harga Rumah Depok")
st.caption(
    "Model regresi dari data listing rumah Depok dan sekitarnya (hasil scraping)."
)

tab_pred, tab_eda, tab_model = st.tabs(["Prediksi", "Eksplorasi Data", "Tentang Model"])

# ---------- Tab 1: Prediksi ----------
with tab_pred:
    kiri, kanan = st.columns([1, 1.3], gap="large")

    with kiri, st.form("form_prediksi"):
        kecamatan = st.selectbox(
            "Lokasi (Kecamatan / Kawasan)",
            meta["kecamatan"],
            index=meta["kecamatan"].index("Cinere"),
        )
        c1, c2, c3 = st.columns(3)
        kt = c1.number_input("Kamar tidur", 1, 15, 3)
        km = c2.number_input("Kamar mandi", 1, 15, 2)
        garasi = c3.number_input("Garasi", 0, 10, 1)
        c4, c5 = st.columns(2)
        lt = c4.number_input("Luas tanah (m²)", 10, 5000, 90)
        lb = c5.number_input("Luas bangunan (m²)", 10, 3000, 70)
        kirim = st.form_submit_button("Prediksi harga", type="primary", width="stretch")

    if kirim:
        x = pd.DataFrame(
            [
                {
                    "kamar_tidur": kt,
                    "kamar_mandi": km,
                    "garasi": garasi,
                    "luas_tanah": lt,
                    "luas_bangunan": lb,
                    "kecamatan": kecamatan,
                }
            ]
        )
        pred = float(model.predict(x)[0])
        mape = M["MAPE"] / 100

        with kiri:
            st.metric("Estimasi harga penawaran", rupiah(pred))
            st.write(
                f"Kisaran wajar (±MAPE {M['MAPE']:.0f}%): **{rupiah(pred * (1 - mape))}** s.d. **{rupiah(pred * (1 + mape))}**"
            )
            if lb > 6 * lt:
                st.warning(
                    "Luas bangunan lebih dari 6× luas tanah, tidak lazim; hasil kurang dapat diandalkan."
                )
            for nama, v in [
                ("kamar_tidur", kt),
                ("kamar_mandi", km),
                ("garasi", garasi),
                ("luas_tanah", lt),
                ("luas_bangunan", lb),
            ]:
                lo, hi = rentang[nama]
                if not lo <= v <= hi:
                    st.info(
                        f"{nama.replace('_', ' ').title()} di luar rentang umum data latih ({lo:.0f}–{hi:.0f})."
                    )

        with kanan:
            sebaran = df[df.kecamatan == kecamatan]
            sebaran = sebaran if len(sebaran) >= 50 else df
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.scatter(
                df.luas_bangunan,
                df.harga_juta,
                s=4,
                alpha=0.12,
                color="gray",
                label="Semua listing",
            )
            ax.scatter(
                sebaran.luas_bangunan,
                sebaran.harga_juta,
                s=7,
                alpha=0.4,
                color="tab:blue",
                label=f"Listing {kecamatan}"
                if len(sebaran) < len(df)
                else "Semua listing",
            )
            ax.scatter(
                [lb], [pred], s=160, marker="*", color="red", zorder=5, label="Prediksi"
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Luas bangunan (m²)")
            ax.set_ylabel("Harga (Juta Rp)")
            ax.set_title("Posisi prediksi dibanding data listing")
            ax.legend(fontsize=8)
            st.pyplot(fig)
    else:
        with kanan:
            st.info("Isi form di kiri lalu klik **Prediksi harga**.")

# ---------- Tab 2: EDA ----------
with tab_eda:
    st.write(
        f"Data bersih yang dipakai model: **{len(df):,}** listing.".replace(",", ".")
    )
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.hist(np.log10(df.harga_juta), bins=50, color="tab:blue")
        ax.set_xlabel("log10(Harga, Juta Rp)")
        ax.set_ylabel("Jumlah listing")
        ax.set_title("Distribusi harga (skala log)")
        st.pyplot(fig)
    with b:
        med = df.groupby("kecamatan").harga_juta.agg(["median", "size"])
        med = med[med["size"] >= 100].sort_values("median")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.barh(med.index, med["median"], color="tab:green")
        ax.set_xlabel("Median harga (Juta Rp)")
        ax.set_title("Median harga per lokasi (≥100 listing)")
        ax.tick_params(axis="y", labelsize=7)
        st.pyplot(fig)
    c, d = st.columns(2)
    with c:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.scatter(df.luas_tanah, df.harga_juta, s=4, alpha=0.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Luas tanah (m²)")
        ax.set_ylabel("Harga (Juta Rp)")
        ax.set_title("Luas tanah vs harga")
        st.pyplot(fig)
    with d:
        m_kt = df[df.kamar_tidur <= 8].groupby("kamar_tidur").harga_juta.median()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(m_kt.index, m_kt.values, marker="o")
        ax.set_xlabel("Jumlah kamar tidur")
        ax.set_ylabel("Median harga (Juta Rp)")
        ax.set_title("Kamar tidur vs median harga")
        st.pyplot(fig)
    with st.expander("Lihat contoh data"):
        st.dataframe(df.head(100), width="stretch")

# ---------- Tab 3: Tentang model ----------
with tab_model:
    st.subheader(f"Model: {meta['model']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", rupiah(M["MAE"]))
    m2.metric("MAPE", f"{M['MAPE']:.1f}%")
    m3.metric("R²", f"{M['R2']:.2f}")
    m4.metric("RMSE", rupiah(M["RMSE"]))
    st.caption("Metrik dihitung pada test set bersih (20% data).")

    st.markdown("**Penjelasan fitur input**")
    st.table(
        pd.DataFrame(
            {
                "Fitur": [
                    "Lokasi",
                    "Kamar tidur",
                    "Kamar mandi",
                    "Garasi",
                    "Luas tanah",
                    "Luas bangunan",
                ],
                "Keterangan": [
                    "Kecamatan / kawasan listing",
                    "Jumlah kamar tidur",
                    "Jumlah kamar mandi",
                    "Jumlah garasi (0 jika tidak ada)",
                    "Luas tanah (m²)",
                    "Luas bangunan (m²)",
                ],
            }
        )
    )
    st.markdown("""
**Alur model:** fitur turunan (rasio luas, total kamar, log luas) → scaling & one-hot lokasi → seleksi fitur
(Mutual Information) → Random Forest dengan target `log1p(harga)`.

**Catatan:** harga adalah *harga penawaran* dari listing, bukan harga transaksi, dan hanya mencakup rumah hunian
di Depok dan sekitarnya. Gunakan hasil sebagai patokan awal, bukan penilaian resmi.
""")
