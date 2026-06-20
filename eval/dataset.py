from pathlib import Path

import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

CACHE_PATH = Path(__file__).parent.parent / "data" / "koi_table.parquet"

_COLUMNS = (
    "kepoi_name,kepid,koi_disposition,koi_pdisposition,"
    "koi_period,koi_time0bk,koi_duration,koi_depth,koi_prad"
)


def load_koi_table() -> pd.DataFrame:
    if CACHE_PATH.exists():
        df = pd.read_parquet(CACHE_PATH)
    else:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        result = NasaExoplanetArchive.query_criteria(
            table="cumulative",
            select=_COLUMNS,
        )
        df = result.to_pandas()
        df.to_parquet(CACHE_PATH, index=False)

    counts = df["koi_disposition"].value_counts()
    print(counts.to_string())
    return df


if __name__ == "__main__":
    load_koi_table()
