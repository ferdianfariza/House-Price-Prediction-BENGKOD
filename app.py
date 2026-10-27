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
    return joblib.load(BASE / "model" / "model_rf.joblib")


@st.cache_resource
def muat_kolom():
    return joblib.load(BASE / "model" / "kolom.joblib")


@st.cache_data
def muat_data():
    return pd.read_csv(BASE / "data_bersih.csv")


@st.cache_data
def muat_meta():
    with open(BASE / "model" / "meta.json") as f:
        return json.load(f)


def rupiah(juta):
    """Format harga (Juta Rupiah) ke teks Indonesia."""
    if juta >= 1000:
        teks = (
            f"{juta / 1000:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        )
        return f"Rp {teks} Miliar"
    return "Rp " + f"{juta:,.0f}".replace(",", ".") + " Juta"


model, kolom, df, meta = muat_model(), muat_kolom(), muat_data(), muat_meta()

M = meta["metrik_test"]
MAPE = M["MAPE_%"]
KECAMATAN = sorted(
    c.replace("kecamatan_", "", 1) for c in kolom if c.startswith("kecamatan_")
)
FITUR_NUM = ["kamar_tidur", "kamar_mandi", "luas_tanah", "luas_bangunan"]
rentang = {c: (df[c].min(), df[c].max()) for c in FITUR_NUM}

st.title("🏠 Prediksi Harga Rumah Depok")
st.caption("Model regresi dari data listing rumah Depok dan sekitarnya")

tab_pred, tab_eda, tab_model = st.tabs(["Prediksi", "Eksplorasi Data", "Tentang Model"])

# ---------- Tab 1: Prediksi ----------
with tab_pred:
    kiri, kanan = st.columns([1, 1.3], gap="large")

    with kiri, st.form("form_prediksi"):
        kecamatan = st.selectbox(
            "Lokasi (Kecamatan / Kawasan)",
            KECAMATAN,
            index=KECAMATAN.index("Cinere") if "Cinere" in KECAMATAN else 0,
        )
        c1, c2 = st.columns(2)
        kt = c1.number_input("Kamar tidur", 1, 15, 3)
        km = c2.number_input("Kamar mandi", 1, 15, 2)
        c4, c5 = st.columns(2)
        lt = c4.number_input("Luas tanah (m²)", 10, 5000, 90)
        lb = c5.number_input("Luas bangunan (m²)", 10, 3000, 70)
        kirim = st.form_submit_button("Prediksi harga", type="primary", width="stretch")

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
        pred = float(model.predict(x)[0]) / 1e6  # model memprediksi Rupiah -> Juta
        mape = MAPE / 100

        with kiri:
            st.metric("Estimasi harga penawaran", rupiah(pred))
            st.write(
                f"Kisaran wajar (±MAPE {MAPE:.0f}%): **{rupiah(pred * (1 - mape))}** s.d. **{rupiah(pred * (1 + mape))}**"
            )
            if lb > 6 * lt:
                st.warning(
                    "Luas bangunan lebih dari 6× luas tanah, tidak lazim; hasil kurang dapat diandalkan."
                )
            for nama, v in [
                ("kamar_tidur", kt),
                ("kamar_mandi", km),
                ("luas_tanah", lt),
                ("luas_bangunan", lb),
            ]:
                lo, hi = rentang[nama]
                if not lo <= v <= hi:
                    st.info(
                        f"{nama.replace('_', ' ').title()} di luar rentang data latih ({lo:.0f}–{hi:.0f})."
                    )

        with kanan:
            sebaran = df[df["kecamatan"] == kecamatan]
            sebaran = sebaran if len(sebaran) >= 50 else df
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.scatter(
                df["luas_bangunan"].to_numpy(),
                df["harga_juta"].to_numpy(),
                s=4,
                alpha=0.12,
                color="gray",
                label="Semua listing",
            )
            ax.scatter(
                sebaran["luas_bangunan"].to_numpy(),
                sebaran["harga_juta"].to_numpy(),
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
        ax.hist(np.log10(df["harga_juta"].to_numpy()), bins=50, color="tab:blue")
        ax.set_xlabel("log10(Harga, Juta Rp)")
        ax.set_ylabel("Jumlah listing")
        ax.set_title("Distribusi harga (skala log)")
        st.pyplot(fig)
    with b:
        med = (
            df.groupby("kecamatan")["harga_juta"]
            .agg(median="median", n="size")
            .reset_index()
            .query("n >= 100")
            .sort_values("median")
        )
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.barh(med["kecamatan"].tolist(), med["median"].to_numpy(), color="tab:green")
        ax.set_xlabel("Median harga (Juta Rp)")
        ax.set_title("Median harga per lokasi (≥100 listing)")
        ax.tick_params(axis="y", labelsize=7)
        st.pyplot(fig)
    c, d = st.columns(2)
    with c:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.scatter(
            df["luas_tanah"].to_numpy(), df["harga_juta"].to_numpy(), s=4, alpha=0.2
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Luas tanah (m²)")
        ax.set_ylabel("Harga (Juta Rp)")
        ax.set_title("Luas tanah vs harga")
        st.pyplot(fig)
    with d:
        m_kt = (
            df[df["kamar_tidur"] <= 8]
            .groupby("kamar_tidur", as_index=False)["harga_juta"]
            .median()
            .sort_values("kamar_tidur")
        )
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(
            m_kt["kamar_tidur"].to_numpy(), m_kt["harga_juta"].to_numpy(), marker="o"
        )
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
    m1.metric("MAE", rupiah(M["MAE"] / 1e6))
    m2.metric("MAPE", f"{MAPE:.1f}%")
    m3.metric("R²", f"{M['R2']:.1f}")
    m4.metric("RMSE", rupiah(M["RMSE"] / 1e6))
    st.caption("Metrik dihitung pada test set bersih (20% data).")

    st.markdown("**Penjelasan fitur input**")
    st.table(
        {
            "Fitur": [
                "Lokasi",
                "Kamar tidur",
                "Kamar mandi",
                "Luas tanah",
                "Luas bangunan",
            ],
            "Keterangan": [
                "Kecamatan / kawasan listing",
                "Jumlah kamar tidur (maks. 5 pada data latih)",
                "Jumlah kamar mandi (maks. 5 pada data latih)",
                "Luas tanah (m²)",
                "Luas bangunan (m²)",
            ],
        }
    )
    st.markdown("""
**Alur model:** one-hot lokasi → Random Forest dengan target `log1p(harga)`.

**Catatan:** harga adalah *harga penawaran* dari listing, bukan harga transaksi, dan hanya mencakup rumah hunian
di Depok dan sekitarnya. Gunakan hasil sebagai patokan awal, bukan penilaian resmi.
""")
