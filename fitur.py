import numpy as np
import pandas as pd

KECAMATAN_RESMI = [
    "Beji",
    "Bojong Sari",
    "Cilodong",
    "Cimanggis",
    "Cinere",
    "Cipayung",
    "Limo",
    "Pancoran Mas",
    "Sawangan",
    "Sukmajaya",
    "Tapos",
]


def add_features(df):
    d = df.copy()
    d["total_kamar"] = d["kamar_tidur"] + d["kamar_mandi"]
    d["ada_garasi"] = (d["garasi"] > 0).astype(int)
    d["rasio_lb_lt"] = (d["luas_bangunan"] / d["luas_tanah"]).replace(
        [np.inf, -np.inf], np.nan
    )
    d["log_luas_tanah"] = np.log1p(d["luas_tanah"])
    d["log_luas_bangunan"] = np.log1p(d["luas_bangunan"])
    d["kecamatan_resmi"] = d["kecamatan"].isin(KECAMATAN_RESMI).astype(int)
    return d
