import warnings
from typing import TypedDict

import numpy as np
from astropy.timeseries import BoxLeastSquares
from lightkurve import LightCurve


class BLSResult(TypedDict):
    period: float    # days
    t0: float        # BKJD
    duration: float  # days
    depth: float     # fractional flux drop
    power: float     # peak BLS power


def bls_search(
    lc: LightCurve,
    min_period: float,
    max_period: float,
    n_periods: int = 2000,
    durations: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3),
) -> BLSResult:
    t = lc.time.value
    f = lc.flux.value

    periods = np.linspace(min_period, max_period, n_periods)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BoxLeastSquares(t, f)
        result = model.power(periods, durations)

    best = int(np.argmax(result.power))
    return BLSResult(
        period=float(result.period[best]),
        t0=float(result.transit_time[best]),
        duration=float(result.duration[best]),
        depth=float(result.depth[best]),
        power=float(result.power[best]),
    )
