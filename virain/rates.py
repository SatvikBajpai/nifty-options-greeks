"""Risk-free rate input.

Deliberately a thin, swappable layer. The rate matters far less than it looks:
the forward is extracted from the chain using the SAME discount factor that
prices the option, so a mis-specified r is largely absorbed into F and mostly
cancels out of the implied vol. See tools/rate_sensitivity.py for the measured
size of that effect.

Supply a real curve by dropping a CSV at data/rates.csv with columns
`date,rate` (rate as a decimal, e.g. 0.0672 - 91-day T-bill or MIBOR both work).
Missing dates are forward-filled, so a monthly series is fine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA, DEFAULT_RATE

CSV = DATA / "rates.csv"


def load(dates: pd.Series, default: float = DEFAULT_RATE) -> pd.Series:
    """Return an annualised continuous rate aligned to `dates`."""
    if not CSV.exists():
        return pd.Series(np.full(len(dates), default), index=dates.index)
    curve = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date")
    r = curve["rate"].to_numpy(float)
    if np.nanmedian(r) > 1.0:          # tolerate percent-quoted files
        r = r / 100.0
    curve["rate"] = r
    out = pd.merge_asof(
        pd.DataFrame({"date": pd.to_datetime(dates.values)}).sort_values("date"),
        curve, on="date", direction="backward",
    )
    return pd.Series(out["rate"].fillna(default).to_numpy(), index=dates.index)
