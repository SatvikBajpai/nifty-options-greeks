"""Paths and constants for the virain NSE options dataset."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("VIRAIN_DATA", ROOT / "data"))

RAW_FO = DATA / "raw" / "fo"      # bhavcopy zips, exactly as NSE serves them
FO = DATA / "fo"                  # normalised per-day parquet
SPOT = DATA / "spot"              # index OHLC
CHAIN = DATA / "chain"            # final dataset with IV + greeks

# NSE switched the F&O bhavcopy to the UDiFF schema on 2024-01-01.
# Legacy `foDDMMMYYYYbhav.csv.zip` files stop being served after 2024-07-05,
# UDiFF files do not exist before 2024-01-01, so the boundary is clean.
UDIFF_START = date(2024, 1, 1)

BASE = "https://nsearchives.nseindia.com"

# NSE 403s an empty User-Agent but does not require cookies for the archive host.
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# Symbols kept when normalising a bhavcopy. The raw zips are archived in full,
# so widening this later costs a re-parse, never a re-download.
DEFAULT_SYMBOLS = ("NIFTY",)

# Flat fallback risk-free rate (annualised, continuous). Only used when the
# put-call-parity fit cannot recover a discount factor and no rates file exists.
DEFAULT_RATE = 0.065

DAYS_PER_YEAR = 365.0

for _p in (RAW_FO, FO, SPOT, CHAIN):
    _p.mkdir(parents=True, exist_ok=True)
