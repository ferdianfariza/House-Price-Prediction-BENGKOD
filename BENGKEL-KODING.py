import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project Early-Bird

    Ferdian Nur Fariza
    """)
    return


@app.cell
def _():
    import re

    import altair as alt
    import marimo as mo
    import polars as pl

    return alt, mo, pl, re


@app.cell
def _(pl):
    df = pl.read_csv("./Raw-Final-Rumah.csv", infer_schema=False)
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df.null_count()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. EDA - Temuan awal pada data mentah

    Dari sampel data terlihat:

    1. **Kolom bergeser ke kiri.** Jika Garasi/Kamar tidak ada, nilai sesudahnya ikut maju
       (contoh: `525 Juta,2,1,LB:41 m²,,,Cinere`). Isi kolom tidak sesuai namanya.
    2. **Harga berakhiran `Total`** (contoh `30 Miliar Total`) biasanya listing tanah/kavling
       tanpa kamar dan bangunan. Ini bukan rumah, jadi maknanya berbeda.
    3. **Duplikat tersembunyi.** Listing yang sama (`385 Juta, 3, 2, 1, 50, 60`) muncul
       dengan `Kecamatan` dan `Page` berbeda, sehingga `is_duplicated()` pada semua kolom tidak menangkapnya.
    4. **Kecamatan tidak setingkat**: `Depok I`/`Depok II` (sub-wilayah), `Margonda` (jalan/kelurahan),
       `Cibubur` (kawasan, bukan kecamatan).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.a Parsing Kolom Harga

    2. **Harga berakhiran `Total`** (contoh `30 Miliar Total`) biasanya listing tanah/kavling
       tanpa kamar dan bangunan. Ini bukan rumah, jadi maknanya berbeda.
    """)
    return


@app.cell
def _(re):
    def to_num(s):
        """'1.368' / '1,5' / '438' -> float. Titik 3 digit = pemisah ribuan."""
        if s is None:
            return None
        s = s.strip()
        if not s:
            return None
        if "." in s and "," in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        elif re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
            s = s.replace(".", "")
        try:
            return float(s)
        except ValueError:
            return None

    UNIT = {
        "triliun": 1e12,
        "miliar": 1e9,
        "milyar": 1e9,
        "juta": 1e6,
        "ribu": 1e3,
    }

    def parse_harga(s):
        """Return (harga_rupiah, is_total, unit_ditemukan)."""
        if s is None:
            return None, False, None
        t = s.strip()
        is_total = "total" in t.lower()
        m = re.search(
            r"([\d.,]+)\s*(triliun|miliar|milyar|juta|ribu)?", t, re.I
        )
        if not m:
            return None, is_total, None
        val = to_num(m.group(1))
        unit = (m.group(2) or "").lower()
        if val is None:
            return None, is_total, unit or None
        return val * UNIT.get(unit, 1), is_total, unit or None

    def parse_row(row):
        """Rapikan kolom yang bergeser: kumpulkan semua sel lalu klasifikasi isinya."""
        cells = [
            row[c]
            for c in [
                "Kamar Tidur",
                "Kamar Mandi",
                "Garasi",
                "Luas Tanah",
                "Luas Bangunan",
            ]
        ]
        lt = lb = None
        nums = []
        shifted = False
        for i, c in enumerate(cells):
            if c is None or not c.strip():
                continue
            c = c.strip()
            m = re.match(r"^(LT|LB)\s*:\s*([\d.,]+)", c, re.I)
            if m:
                if m.group(1).upper() == "LT":
                    lt = to_num(m.group(2))
                    shifted |= i != 3
                else:
                    lb = to_num(m.group(2))
                    shifted |= i != 4
            else:
                nums.append(to_num(c))
        # urutan angka murni: Kamar Tidur, Kamar Mandi, Garasi
        kt = nums[0] if len(nums) > 0 else None
        km = nums[1] if len(nums) > 1 else None
        gr = nums[2] if len(nums) > 2 else None
        harga, is_total, unit = parse_harga(row["Harga"])
        return {
            "harga": harga,
            "harga_total": is_total,
            "unit_harga": unit,
            "kamar_tidur": kt,
            "kamar_mandi": km,
            "garasi": gr,
            "luas_tanah": lt,
            "luas_bangunan": lb,
            "kecamatan": (row["Kecamatan"] or "").strip() or None,
            "page": int(row["Page"])
            if row["Page"] and row["Page"].isdigit()
            else None,
            "kolom_bergeser": shifted,
        }

    return (parse_row,)


@app.cell
def _(df, parse_row, pl):
    parsed = pl.DataFrame(
        [parse_row(r) for r in df.iter_rows(named=True)],
        infer_schema_length=None,
    ).with_columns(
        # listing tanah/kavling: tidak punya kamar & bangunan
        (
            pl.col("kamar_tidur").is_null()
            & pl.col("kamar_mandi").is_null()
            & pl.col("luas_bangunan").is_null()
        ).alias("tanpa_bangunan"),
        pl.col("harga").log10().alias("log_harga"),
    )
    parsed.head(10)
    return (parsed,)


@app.cell
def _(parsed):
    parsed.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tabel temuan kualitas data
    """)
    return


@app.cell
def _(df, parsed, pl):
    _key = [
        "harga",
        "kamar_tidur",
        "kamar_mandi",
        "garasi",
        "luas_tanah",
        "luas_bangunan",
    ]
    _dup_exact = df.is_duplicated().sum()
    _dup_listing = parsed.select(_key).is_duplicated().sum()

    quality = pl.DataFrame(
        {
            "masalah": [
                "Duplikat identik (semua kolom)",
                "Duplikat listing (abaikan Kecamatan & Page)",
                "Kolom bergeser (LT/LB bukan di kolomnya)",
                "Harga bertanda 'Total'",
                "Listing tanpa bangunan (tanah/kavling)",
                "Harga gagal diparse",
                "Kamar tidur kosong",
                "Kamar mandi kosong",
                "Garasi kosong",
                "Luas tanah kosong",
                "Luas bangunan kosong",
                "Kecamatan kosong",
            ],
            "jumlah_baris": [
                _dup_exact,
                _dup_listing,
                parsed["kolom_bergeser"].sum(),
                parsed["harga_total"].sum(),
                parsed["tanpa_bangunan"].sum(),
                parsed["harga"].null_count(),
                parsed["kamar_tidur"].null_count(),
                parsed["kamar_mandi"].null_count(),
                parsed["garasi"].null_count(),
                parsed["luas_tanah"].null_count(),
                parsed["luas_bangunan"].null_count(),
                parsed["kecamatan"].null_count(),
            ],
            "kemungkinan_penyebab": [
                "Listing tampil ulang saat scraping",
                "Satu listing dipasang di banyak Kecamatan/halaman",
                "Elemen HTML opsional (garasi/kamar) tidak ada, nilai ter-shift",
                "Listing tanah/kavling dengan harga seluruh lahan",
                "Tidak ada info kamar & luas bangunan di sumber",
                "Format harga tidak terduga",
                "Listing tanah / data tidak diisi",
                "Listing tanah / data tidak diisi",
                "Garasi tidak ditulis (bisa berarti 0, perlu validasi)",
                "Ikut bergeser atau tidak ditulis",
                "Listing tanah atau tidak ditulis",
                "Gagal scraping",
            ],
        }
    )
    quality
    return


@app.cell
def _(parsed):
    # Variasi satuan harga yang muncul
    parsed.group_by("unit_harga").len().sort("len", descending=True)
    return


@app.cell
def _(parsed):
    # Kecamatan: cek tingkat wilayah & ketidakmerataan jumlah data
    parsed.group_by("kecamatan").len().sort("len", descending=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualisasi
    """)
    return


@app.cell
def _(alt, parsed, pl):
    # Distribusi harga: miring ke kanan -> pakai skala log.
    # Binning dihitung di polars (lebar 0.1 pada skala log10) supaya Altair hanya
    # menerima puluhan baris, bukan puluhan ribu (batas default Altair 5000 baris).
    _lebar = 0.1
    _hist = (
        parsed.filter(~pl.col("tanpa_bangunan"))
        .drop_nulls("log_harga")
        .with_columns(
            ((pl.col("log_harga") / _lebar).floor() * _lebar).alias("bin_awal")
        )
        .group_by("bin_awal")
        .len()
        .rename({"len": "jumlah"})
        .with_columns((pl.col("bin_awal") + _lebar).alias("bin_akhir"))
        .sort("bin_awal")
    )
    alt.Chart(_hist).mark_bar().encode(
        alt.X("bin_awal:Q", title="log10(Harga)"),
        alt.X2("bin_akhir:Q"),
        alt.Y("jumlah:Q", title="Jumlah listing"),
        tooltip=["bin_awal", "bin_akhir", "jumlah"],
    ).properties(
        title="Distribusi log10(Harga) - rumah", width=500, height=300
    )
    return


@app.cell
def _(alt, parsed):
    # Harga vs luas bangunan (skala log)
    _d = parsed.filter(~parsed["tanpa_bangunan"]).drop_nulls(
        ["harga", "luas_bangunan"]
    )
    alt.Chart(_d.sample(min(5000, _d.height), seed=42)).mark_circle(
        opacity=0.3
    ).encode(
        alt.X(
            "luas_bangunan:Q",
            scale=alt.Scale(type="log"),
            title="Luas bangunan (m²)",
        ),
        alt.Y("harga:Q", scale=alt.Scale(type="log"), title="Harga (Rp)"),
    ).properties(title="Harga vs Luas Bangunan", width=500, height=300)
    return


@app.cell
def _(alt, parsed, pl):
    # Median harga per Kecamatan (hanya yang datanya cukup)
    _agg = (
        parsed.filter(~pl.col("tanpa_bangunan"))
        .group_by("kecamatan")
        .agg(
            pl.col("harga").median().alias("median_harga"), pl.len().alias("n")
        )
        .filter(pl.col("n") >= 30)
        .sort("median_harga", descending=True)
    )
    alt.Chart(_agg).mark_bar().encode(
        alt.X("median_harga:Q", title="Median harga (Rp)"),
        alt.Y("kecamatan:N", sort="-x"),
        tooltip=["kecamatan", "median_harga", "n"],
    ).properties(title="Median Harga per Kecamatan (n >= 30)", width=500)
    return


@app.cell
def _(parsed, pl):
    # Korelasi (Spearman) fitur numerik dengan harga, tahan terhadap outlier
    _cols = [
        "kamar_tidur",
        "kamar_mandi",
        "garasi",
        "luas_tanah",
        "luas_bangunan",
        "page",
    ]
    _d = parsed.filter(~pl.col("tanpa_bangunan")).drop_nulls(["harga"])
    pl.DataFrame(
        {
            "fitur": _cols,
            "spearman_vs_harga": [
                _d.select(pl.corr(c, "harga", method="spearman")).item()
                for c in _cols
            ],
        }
    ).sort("spearman_vs_harga", descending=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Insight (isi setelah melihat output)

    1. **Distribusi harga miring kanan** -> transformasi `log` pada target (atau `TransformedTargetRegressor`),
       dan gunakan MAE/MAPE selain R².
    2. **Luas bangunan/tanah paling berhubungan dengan harga**, tetapi ada outlier -> bedakan
       ekstrem valid (mis. 40 Miliar, LT 4000 m²) dari salah input (mis. harga 28,1 Juta untuk LT 12 m²)
       dengan harga per m² dan batas IQR.
    3. **Kecamatan tidak merata & tidak setingkat** -> gabungkan/standarkan wilayah, kelompokkan kategori
       langka jadi `Lainnya`, lalu target/frequency encoding.
    4. **`Page` bukan karakteristik rumah** -> kandidat dibuang (cek korelasinya di atas).
    5. **Listing "Total" / tanah** -> dipisah dari data rumah karena target berbeda maknanya.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Direct Modeling (sebelum preprocessing)

    Kontrol percobaan. Data hanya di-*parse* ke angka (tak terhindarkan karena aslinya teks), tanpa
    membuang duplikat, listing tanah, outlier, atau transformasi target. Nilai kosong hanya diimputasi
    median di dalam pipeline agar semua model bisa jalan. Test set 20% (acak, `random_state=42`).

    **Alasan pemilihan model** (6 model, 2 di antaranya *ensemble*):

    | Model | Alasan |
    |---|---|
    | Linear Regression | Baseline paling sederhana. Menunjukkan seberapa jauh hubungan linear bisa menjelaskan harga. |
    | KNN Regressor | Harga rumah cenderung mirip dengan rumah bertetangga (lokasi & ukuran mirip). Sensitif skala dan outlier, jadi pembeda yang baik sebelum/sesudah preprocessing. |
    | Decision Tree | Menangkap hubungan non-linear dan interaksi (mis. luas × lokasi) tanpa asumsi distribusi. |
    | Random Forest (ensemble, bagging) | Mengurangi varians pohon tunggal, tahan terhadap outlier dan fitur campuran. |
    | Extra Trees (ensemble, bagging acak) | Seperti RF tetapi split lebih acak, biasanya lebih cepat dan lebih halus pada data noisy. |
    | HistGradientBoosting (ensemble, boosting) | Biasanya terbaik pada data tabular, cepat pada ±40 ribu baris, dan bisa menangani nilai kosong. |
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from scipy.stats import loguniform, randint
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        r2_score,
    )
    from sklearn.model_selection import (
        KFold,
        RandomizedSearchCV,
        train_test_split,
    )
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    return (
        ColumnTransformer,
        DecisionTreeRegressor,
        ExtraTreesRegressor,
        HistGradientBoostingRegressor,
        KFold,
        KNeighborsRegressor,
        LinearRegression,
        OneHotEncoder,
        Pipeline,
        RandomForestRegressor,
        RandomizedSearchCV,
        SimpleImputer,
        StandardScaler,
        TransformedTargetRegressor,
        loguniform,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mutual_info_regression,
        np,
        r2_score,
        randint,
        train_test_split,
    )


@app.cell
def _():
    NUM_D = [
        "kamar_tidur",
        "kamar_mandi",
        "garasi",
        "luas_tanah",
        "luas_bangunan",
        "page",
    ]
    NUM_KAND = [
        "kamar_tidur",
        "kamar_mandi",
        "garasi",
        "luas_tanah",
        "luas_bangunan",
        "rasio_bangunan",
        "kamar_total",
        "page",
    ]
    return NUM_D, NUM_KAND


@app.cell
def _(
    ColumnTransformer,
    DecisionTreeRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    KNeighborsRegressor,
    LinearRegression,
    OneHotEncoder,
    Pipeline,
    RandomForestRegressor,
    SimpleImputer,
    StandardScaler,
    TransformedTargetRegressor,
    mean_absolute_error,
    mean_absolute_percentage_error,
    np,
    pl,
    r2_score,
):
    def hitung_metrik(y_true, y_pred):
        return {
            "MAE (juta)": mean_absolute_error(y_true, y_pred) / 1e6,
            "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "R2": r2_score(y_true, y_pred),
        }

    def buat_model():
        return {
            "Linear Regression": LinearRegression(),
            "KNN": KNeighborsRegressor(n_neighbors=10),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(
                n_estimators=200, n_jobs=-1, random_state=42
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=200, n_jobs=-1, random_state=42
            ),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                random_state=42
            ),
        }

    def buat_preprocessor(num, cat, min_freq):
        langkah = []
        if num:
            langkah.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="median")),
                            ("sc", StandardScaler()),
                        ]
                    ),
                    num,
                )
            )
        if cat:
            langkah.append(
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=min_freq,
                        sparse_output=False,
                    ),
                    cat,
                )
            )
        return ColumnTransformer(langkah)

    def buat_pipeline(model, num, cat, log_target, min_freq):
        pipe = Pipeline(
            [("prep", buat_preprocessor(num, cat, min_freq)), ("model", model)]
        )
        if log_target:
            # model belajar pada log(harga), prediksi dikembalikan ke Rupiah
            return TransformedTargetRegressor(
                regressor=pipe, func=np.log1p, inverse_func=np.expm1
            )
        return pipe

    def evaluasi_semua(
        tahap, Xtr, ytr, Xte, yte, num, cat, log_target, min_freq
    ):
        baris, terlatih = [], {}
        for nama, m in buat_model().items():
            p = buat_pipeline(m, num, cat, log_target, min_freq)
            p.fit(Xtr[num + cat], ytr)
            pred = p.predict(Xte[num + cat])
            baris.append(
                {"tahap": tahap, "model": nama, **hitung_metrik(yte, pred)}
            )
            terlatih[nama] = p
        return pl.DataFrame(baris), terlatih

    return buat_pipeline, evaluasi_semua, hitung_metrik


@app.cell
def _(NUM_D, evaluasi_semua, parsed, pl, train_test_split):
    _raw = (
        parsed.filter(pl.col("harga").is_not_null() & (pl.col("harga") > 0))
        .select(NUM_D + ["kecamatan", "harga"])
        .to_pandas()
    )
    _raw["kecamatan"] = _raw["kecamatan"].fillna("Unknown")
    _Xtr, _Xte, _ytr, _yte = train_test_split(
        _raw.drop(columns="harga"),
        _raw["harga"],
        test_size=0.2,
        random_state=42,
    )
    hasil_direct, _ = evaluasi_semua(
        "1. Direct", _Xtr, _ytr, _Xte, _yte, NUM_D, ["kecamatan"], False, 1
    )
    hasil_direct
    return (hasil_direct,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Modeling dengan Preprocessing

    Langkah pembersihan (aturan tetap, tidak bergantung pada distribusi data) dilakukan **sebelum** split.
    Batas outlier berbasis kuantil dihitung **hanya dari train** lalu diterapkan ke train dan test,
    sehingga test set bersih tanpa kebocoran informasi.
    """)
    return


@app.cell
def _(parsed, pl):
    _kunci = [
        "harga",
        "kamar_tidur",
        "kamar_mandi",
        "garasi",
        "luas_tanah",
        "luas_bangunan",
    ]
    _log = [("Data awal", parsed.height)]

    _d = parsed.filter(pl.col("harga").is_not_null() & (pl.col("harga") > 0))
    _log.append(("Buang harga kosong / <= 0", _d.height))

    _d = _d.unique(subset=_kunci, keep="first", maintain_order=True)
    _log.append(
        ("Buang duplikat listing (abaikan Kecamatan & Page)", _d.height)
    )

    _d = _d.filter(~pl.col("tanpa_bangunan") & ~pl.col("harga_total"))
    _log.append(("Buang listing tanah / harga 'Total'", _d.height))

    _d = _d.drop_nulls(["luas_tanah", "luas_bangunan"])
    _log.append(("Buang luas tanah/bangunan kosong", _d.height))

    _d = _d.filter(
        (pl.col("harga") >= 100e6)
        & (pl.col("luas_tanah") >= 15)
        & (pl.col("luas_bangunan") >= 15)
        & (pl.col("luas_bangunan") / pl.col("luas_tanah") <= 5)
        & (pl.col("kamar_tidur").fill_null(0) <= 15)
        & (pl.col("kamar_mandi").fill_null(0) <= 15)
        & (pl.col("garasi").fill_null(0) <= 10)
    )
    _log.append(
        (
            "Buang salah input (harga < 100 jt, luas < 15 m², LB/LT > 5, kamar > 15, garasi > 10)",
            _d.height,
        )
    )

    bersih_awal = _d
    log_pembersihan = pl.DataFrame(
        {
            "langkah": [x[0] for x in _log],
            "sisa_baris": [x[1] for x in _log],
        }
    ).with_columns(
        (-pl.col("sisa_baris").diff()).fill_null(0).alias("baris_dibuang")
    )
    log_pembersihan
    return (bersih_awal,)


@app.cell
def _(bersih_awal, np, pl, train_test_split):
    _d = bersih_awal.with_columns(
        pl.col("garasi").fill_null(
            0
        ),  # asumsi: garasi tidak ditulis = tidak ada
        pl.col("kecamatan")
        .fill_null("Unknown")
        .str.strip_chars()
        .str.to_titlecase(),
        (pl.col("luas_bangunan") / pl.col("luas_tanah")).alias(
            "rasio_bangunan"
        ),
        pl.sum_horizontal("kamar_tidur", "kamar_mandi").alias("kamar_total"),
        (pl.col("harga") / pl.col("luas_bangunan")).alias(
            "harga_per_m2"
        ),  # hanya utk filter outlier
    ).to_pandas()

    _tr, _te = train_test_split(_d, test_size=0.2, random_state=42)

    _lg = np.log10(_tr["harga_per_m2"])
    _q1, _q3 = _lg.quantile([0.25, 0.75])
    _lo, _hi = _q1 - 3 * (_q3 - _q1), _q3 + 3 * (_q3 - _q1)

    def _wajar(frame):
        v = np.log10(frame["harga_per_m2"])
        return (v >= _lo) & (v <= _hi)

    train_c = _tr[_wajar(_tr)].reset_index(drop=True)
    test_c = _te[_wajar(_te)].reset_index(drop=True)
    batas_harga_per_m2 = (10**_lo, 10**_hi)
    ringkasan_split = {
        "train_sebelum": len(_tr),
        "train_sesudah": len(train_c),
        "test_sebelum": len(_te),
        "test_sesudah": len(test_c),
        "batas_harga_per_m2": batas_harga_per_m2,
    }
    ringkasan_split
    return test_c, train_c


@app.cell
def _(NUM_KAND, evaluasi_semua, test_c, train_c):
    hasil_prep, _ = evaluasi_semua(
        "2. Preprocessing",
        train_c,
        train_c["harga"],
        test_c,
        test_c["harga"],
        NUM_KAND,
        ["kecamatan"],
        True,
        30,
    )
    hasil_prep
    return (hasil_prep,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Feature Selection

    Tiga sudut pandang dihitung **pada train saja**: korelasi Spearman, *mutual information*, dan
    *feature importance* Random Forest (target `log(harga)`). Kecamatan diubah sementara menjadi
    median harga per kecamatan agar bisa dinilai sebagai satu fitur numerik.

    Aturan seleksi: pertahankan fitur dengan importance RF ≥ 0.02, lalu jika dua fitur
    terpilih saling berkorelasi > 0.9, buang yang importance-nya lebih kecil (redundan).
    """)
    return


@app.cell
def _(
    NUM_KAND,
    RandomForestRegressor,
    mutual_info_regression,
    np,
    pl,
    train_c,
):
    _y = np.log1p(train_c["harga"])
    _X = train_c[NUM_KAND].copy()
    _X["kecamatan"] = train_c.groupby("kecamatan")["harga"].transform("median")
    _X = _X.fillna(_X.median())

    _rf = RandomForestRegressor(
        n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=42
    ).fit(_X, _y)
    _mi = mutual_info_regression(_X, _y, random_state=42)

    skor_fitur = pl.DataFrame(
        {
            "fitur": list(_X.columns),
            "spearman_vs_harga": [
                _X[c].corr(train_c["harga"], method="spearman")
                for c in _X.columns
            ],
            "mutual_info": _mi,
            "rf_importance": _rf.feature_importances_,
        }
    ).sort("rf_importance", descending=True)
    corr_fitur = _X.corr(method="spearman").abs()
    skor_fitur
    return corr_fitur, skor_fitur


@app.cell
def _(corr_fitur, mo, skor_fitur):
    AMBANG_RF = 0.02
    _imp = dict(zip(skor_fitur["fitur"], skor_fitur["rf_importance"]))
    _lolos = [f for f in skor_fitur["fitur"] if _imp[f] >= AMBANG_RF]
    _buang = set()
    for _i, _a in enumerate(_lolos):
        for _b in _lolos[_i + 1 :]:
            if _a in _buang or _b in _buang:
                continue
            if corr_fitur.loc[_a, _b] > 0.9:
                _buang.add(_a if _imp[_a] < _imp[_b] else _b)

    FITUR_TERPILIH = [f for f in _lolos if f not in _buang]
    FITUR_NUM = [f for f in FITUR_TERPILIH if f != "kecamatan"]
    FITUR_CAT = ["kecamatan"] if "kecamatan" in FITUR_TERPILIH else []
    _tidak = [f for f in skor_fitur["fitur"] if f not in FITUR_TERPILIH]
    mo.md(
        f"**Fitur terpilih:** {', '.join(FITUR_TERPILIH)}  \n"
        f"**Dibuang:** {', '.join(_tidak) or '-'}"
    )
    return FITUR_CAT, FITUR_NUM


@app.cell
def _(FITUR_CAT, FITUR_NUM, evaluasi_semua, test_c, train_c):
    hasil_fs, _ = evaluasi_semua(
        "3. Feature Selection",
        train_c,
        train_c["harga"],
        test_c,
        test_c["harga"],
        FITUR_NUM,
        FITUR_CAT,
        True,
        30,
    )
    hasil_fs
    return (hasil_fs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Hyperparameter Tuning

    Tiga model berbasis pohon (Random Forest, Extra Trees, HistGradientBoosting) dituning dengan
    `RandomizedSearchCV` (20 kombinasi, 5-fold CV, skor = MAE) **pada data latih saja**.
    Test set bersih hanya dipakai sekali di akhir untuk evaluasi.
    Model lain tidak dituning karena Linear/KNN/Decision Tree umumnya kalah jauh dari ensemble pada data seperti ini.
    """)
    return


@app.cell
def _(mo):
    tombol_tuning = mo.ui.run_button(
        label="Jalankan Hyperparameter Tuning (beberapa menit)"
    )
    tombol_tuning
    return (tombol_tuning,)


@app.cell
def _(
    ExtraTreesRegressor,
    FITUR_CAT,
    FITUR_NUM,
    HistGradientBoostingRegressor,
    KFold,
    RandomForestRegressor,
    RandomizedSearchCV,
    buat_pipeline,
    hitung_metrik,
    loguniform,
    mo,
    pl,
    randint,
    test_c,
    tombol_tuning,
    train_c,
):
    mo.stop(
        not tombol_tuning.value,
        mo.md("_Klik tombol di atas untuk menjalankan tuning._"),
    )

    _p = "regressor__model__"
    _pohon = {
        _p + "n_estimators": randint(100, 400),
        _p + "max_depth": [None, 10, 20, 30],
        _p + "min_samples_leaf": randint(1, 6),
        _p + "max_features": [0.5, 0.7, 1.0],
    }
    _ruang = {
        "Random Forest": (
            RandomForestRegressor(random_state=42, n_jobs=1),
            _pohon,
        ),
        "Extra Trees": (
            ExtraTreesRegressor(random_state=42, n_jobs=1),
            _pohon,
        ),
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(random_state=42),
            {
                _p + "learning_rate": loguniform(0.02, 0.3),
                _p + "max_iter": randint(200, 800),
                _p + "max_leaf_nodes": randint(15, 63),
                _p + "min_samples_leaf": randint(10, 60),
                _p + "l2_regularization": loguniform(1e-3, 10),
            },
        ),
    }

    _Xtr = train_c[FITUR_NUM + FITUR_CAT]
    _Xte = test_c[FITUR_NUM + FITUR_CAT]
    _baris, _param = [], []
    model_tuned = {}
    for _nama, (_m, _dist) in _ruang.items():
        _cari = RandomizedSearchCV(
            buat_pipeline(_m, FITUR_NUM, FITUR_CAT, True, 30),
            _dist,
            n_iter=20,
            cv=KFold(5, shuffle=True, random_state=42),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=42,
        ).fit(_Xtr, train_c["harga"])
        model_tuned[_nama] = _cari.best_estimator_
        _baris.append(
            {
                "tahap": "4. Tuned",
                "model": _nama,
                **hitung_metrik(test_c["harga"], _cari.predict(_Xte)),
            }
        )
        _param.append(
            {
                "model": _nama,
                "cv_mae_juta": -_cari.best_score_ / 1e6,
                "param_terbaik": str(
                    {
                        k.replace(_p, ""): v
                        for k, v in _cari.best_params_.items()
                    }
                ),
            }
        )

    tabel_tuning = pl.DataFrame(_baris)
    param_terbaik = pl.DataFrame(_param)
    param_terbaik
    return model_tuned, tabel_tuning


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Perbandingan Seluruh Tahap

    Catatan: tahap 1 dievaluasi pada test set mentah (masih ada outlier/duplikat), tahap 2-4 pada test set bersih
    yang sama. Jadi tahap 1 adalah kontrol "tanpa preprocessing", bukan perbandingan angka yang sepenuhnya setara.
    """)
    return


@app.cell
def _(hasil_direct, hasil_fs, hasil_prep, pl, tabel_tuning):
    semua_hasil = pl.concat([hasil_direct, hasil_prep, hasil_fs, tabel_tuning])
    semua_hasil
    return (semua_hasil,)


@app.cell
def _(alt, semua_hasil):
    alt.Chart(semua_hasil).mark_bar().encode(
        alt.X("model:N", title=None),
        alt.Y("MAE (juta):Q", title="MAE (juta Rp)"),
        alt.XOffset("tahap:N"),
        alt.Color("tahap:N"),
        tooltip=["tahap", "model", "MAE (juta)", "MAPE (%)", "R2"],
    ).properties(
        title="MAE per Model dan Tahap (lebih kecil lebih baik)",
        width=600,
        height=320,
    )
    return


@app.cell
def _(mo, model_tuned, tabel_tuning):
    _t = tabel_tuning.sort("MAE (juta)").row(0, named=True)
    nama_model_terbaik = _t["model"]
    model_terbaik = model_tuned[nama_model_terbaik]
    mo.md(
        f"**Model terbaik (test set bersih): {nama_model_terbaik}**  \n"
        f"MAE = Rp {_t['MAE (juta)']:.1f} juta, MAPE = {_t['MAPE (%)']:.1f}%, R² = {_t['R2']:.3f}"
    )
    return


if __name__ == "__main__":
    app.run()
