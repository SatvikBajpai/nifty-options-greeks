"""NIFTY 50 daily OHLC (and India VIX, used later as a sanity check on our IV)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from .config import SPOT

TICKERS = {"NIFTY": "^NSEI", "INDIAVIX": "^INDIAVIX"}


def _path(name: str) -> "object":
    return SPOT / f"{name.lower()}_daily.parquet"


def download(name: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    ticker = TICKERS[name]
    raw = yf.download(ticker, start=start, end=end + pd.Timedelta(days=1),
                      auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned nothing for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = (raw.rename(columns=str.lower)
             .reset_index()
             .rename(columns={"Date": "date", "index": "date"}))
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    cols = [c for c in ("date", "open", "high", "low", "close", "volume") if c in df]
    df = df[cols].dropna(subset=["close"]).reset_index(drop=True)
    p = _path(name)
    df.to_parquet(p, index=False)
    df.to_csv(p.with_suffix(".csv"), index=False)
    return df


def load(name: str = "NIFTY") -> pd.DataFrame:
    p = _path(name)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run the download step first")
    return pd.read_parquet(p)
