import warnings

import lightkurve as lk
from lightkurve import LightCurve


def fetch_lightcurve(kepid: int) -> LightCurve:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = lk.search_lightcurve(
            f"KIC {kepid}",
            mission="Kepler",
            author="Kepler",
            exptime="long",
        )

    if len(results) == 0:
        raise ValueError(f"No Kepler light curves found for KIC {kepid}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lcc = results.download_all(quality_bitmask="default")

    # stitch normalizes each quarter before concatenating by default
    lc = lcc.stitch()
    lc = lc.remove_nans()
    lc = lc.normalize()
    return lc
