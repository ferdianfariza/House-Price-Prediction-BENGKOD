import polars as pl

KECAMATAN_RESMI = ['Beji', 'Bojong Sari', 'Cilodong', 'Cimanggis', 'Cinere', 'Cipayung',
                   'Limo', 'Pancoran Mas', 'Sawangan', 'Sukmajaya', 'Tapos']

NUM = ['kamar_tidur', 'kamar_mandi', 'garasi', 'luas_tanah', 'luas_bangunan', 'total_kamar', 'ada_garasi',
       'rasio_lb_lt', 'log_luas_tanah', 'log_luas_bangunan', 'kecamatan_resmi']


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col('kamar_tidur') + pl.col('kamar_mandi')).alias('total_kamar'),
        (pl.col('garasi') > 0).cast(pl.Int64).alias('ada_garasi'),
        pl.when(pl.col('luas_tanah') > 0).then(pl.col('luas_bangunan') / pl.col('luas_tanah')).alias('rasio_lb_lt'),
        pl.col('luas_tanah').log1p().alias('log_luas_tanah'),
        pl.col('luas_bangunan').log1p().alias('log_luas_bangunan'),
        pl.col('kecamatan').is_in(KECAMATAN_RESMI).cast(pl.Int64).alias('kecamatan_resmi'),
    )


def ke_matriks(df: pl.DataFrame):
    """6 kolom bersih (polars) -> matriks fitur [NUM..., kecamatan] untuk scikit-learn."""
    return add_features(df).select(NUM + ['kecamatan']).to_numpy()
