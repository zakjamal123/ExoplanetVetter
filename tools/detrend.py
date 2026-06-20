import numpy as np
from lightkurve import LightCurve


def detrend(lc: LightCurve, window_length: int = 401) -> LightCurve:
    # window_length must be odd for Savitzky-Golay
    if window_length % 2 == 0:
        window_length += 1
    flat = lc.flatten(window_length=window_length, polyorder=2, niters=3, sigma=3)
    # Remove residual upward outliers (cosmic rays / flares).
    # sigma_lower=inf means we NEVER clip downward, so transits are untouched.
    flat = flat.remove_outliers(sigma_upper=4, sigma_lower=np.inf)
    return flat