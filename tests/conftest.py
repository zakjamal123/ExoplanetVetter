import pytest

from tools.detrend import detrend
from tools.fetch import fetch_lightcurve

_KEPID = 11446443


@pytest.fixture(scope="session")
def flat_lc():
    lc = fetch_lightcurve(_KEPID)
    return detrend(lc)
