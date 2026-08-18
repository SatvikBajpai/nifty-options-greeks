"""Download and normalise the NSE F&O bhavcopy (both schemas).

Two file formats exist over a 5-year window:

  legacy  <= 2023-12-29  /content/historical/DERIVATIVES/2023/AUG/fo14AUG2023bhav.csv.zip
  UDiFF   >= 2024-01-01  /content/fo/BhavCopy_NSE_FO_0_0_0_20250814_F_0000.csv.zip

Both are normalised onto one schema (see COLUMNS). Raw zips are cached on disk,
so a re-parse never re-downloads, and a 404 (market holiday) is memoised with a
`.missing` marker so reruns skip it.
"""
from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from .config import BASE, FO, HTTP_HEADERS, RAW_FO, UDIFF_START, DEFAULT_SYMBOLS

COLUMNS = [
    "date", "symbol", "instrument", "expiry", "strike", "opt_type",
    "open", "high", "low", "close", "settle",
    "contracts", "oi", "chg_oi", "lot_size", "underlying",
]

# UDiFF FinInstrmTp -> legacy INSTRUMENT
_UDIFF_INSTR = {"IDF": "FUTIDX", "IDO": "OPTIDX", "STF": "FUTSTK", "STO": "OPTSTK"}


def url_for(d: date) -> str:
    if d >= UDIFF_START:
        return f"{BASE}/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    mon = d.strftime("%b").upper()
    return f"{BASE}/content/historical/DERIVATIVES/{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip"


def _zip_path(d: date) -> Path:
    return RAW_FO / f"{d:%Y}" / f"fo_{d:%Y%m%d}.zip"


def fetch(d: date, session: requests.Session | None = None, retries: int = 3) -> bytes | None:
    """Return the zip bytes for one trading day, or None if NSE has no file (holiday)."""
    path = _zip_path(d)
    if path.exists():
        return path.read_bytes()
    if path.with_suffix(".missing").exists():
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()
    last = None
    for attempt in range(retries):
        try:
            r = sess.get(url_for(d), headers=HTTP_HEADERS, timeout=45)
        except requests.RequestException as exc:
            last = exc
            continue
        if r.status_code == 404:
            path.with_suffix(".missing").touch()
            return None
        if r.status_code == 200 and r.content[:2] == b"PK":
            # write-then-rename: a partially written zip must never be visible
            # under the cache path, or a later run reads a truncated file.
            tmp = path.with_suffix(".zip.part")
            tmp.write_bytes(r.content)
            tmp.replace(path)
            return r.content
        last = RuntimeError(f"HTTP {r.status_code} for {d}")
    raise RuntimeError(f"failed to fetch bhavcopy for {d}: {last}")


def verify_cache(delete_bad: bool = True) -> list:
    """Return (and optionally delete) cached zips that no longer open."""
    bad = []
    for z in sorted(RAW_FO.glob("*/fo_*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                if zf.testzip() is not None:
                    raise zipfile.BadZipFile("crc mismatch")
        except Exception as exc:
            bad.append((z, repr(exc)))
            if delete_bad:
                z.unlink()
    return bad


def _read_csv(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as fh:
            return pd.read_csv(fh, low_memory=False)


def _normalise_udiff(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["TradDt"]),
        "symbol": raw["TckrSymb"].astype("string").str.strip(),
        "instrument": raw["FinInstrmTp"].astype("string").str.strip().map(_UDIFF_INSTR),
        "expiry": pd.to_datetime(raw["XpryDt"]),
        "strike": pd.to_numeric(raw["StrkPric"], errors="coerce"),
        "opt_type": raw["OptnTp"].astype("string").str.strip().fillna("FUT"),
        "open": pd.to_numeric(raw["OpnPric"], errors="coerce"),
        "high": pd.to_numeric(raw["HghPric"], errors="coerce"),
        "low": pd.to_numeric(raw["LwPric"], errors="coerce"),
        "close": pd.to_numeric(raw["ClsPric"], errors="coerce"),
        "settle": pd.to_numeric(raw["SttlmPric"], errors="coerce"),
        "contracts": pd.to_numeric(raw["TtlTradgVol"], errors="coerce"),
        "oi": pd.to_numeric(raw["OpnIntrst"], errors="coerce"),
        "chg_oi": pd.to_numeric(raw["ChngInOpnIntrst"], errors="coerce"),
        "lot_size": pd.to_numeric(raw["NewBrdLotQty"], errors="coerce"),
        "underlying": pd.to_numeric(raw["UndrlygPric"], errors="coerce"),
    })
    return df[df["instrument"].notna()]


def _normalise_legacy(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.rename(columns=lambda c: str(c).strip().upper())
    opt = raw["OPTION_TYP"].astype("string").str.strip().replace({"XX": "FUT"})
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["TIMESTAMP"], format="%d-%b-%Y"),
        "symbol": raw["SYMBOL"].astype("string").str.strip(),
        "instrument": raw["INSTRUMENT"].astype("string").str.strip(),
        "expiry": pd.to_datetime(raw["EXPIRY_DT"], format="%d-%b-%Y"),
        "strike": pd.to_numeric(raw["STRIKE_PR"], errors="coerce"),
        "opt_type": opt,
        "open": pd.to_numeric(raw["OPEN"], errors="coerce"),
        "high": pd.to_numeric(raw["HIGH"], errors="coerce"),
        "low": pd.to_numeric(raw["LOW"], errors="coerce"),
        "close": pd.to_numeric(raw["CLOSE"], errors="coerce"),
        "settle": pd.to_numeric(raw["SETTLE_PR"], errors="coerce"),
        "contracts": pd.to_numeric(raw["CONTRACTS"], errors="coerce"),
        "oi": pd.to_numeric(raw["OPEN_INT"], errors="coerce"),
        "chg_oi": pd.to_numeric(raw["CHG_IN_OI"], errors="coerce"),
        # legacy bhavcopy carries neither lot size nor the underlying print
        "lot_size": pd.NA,
        "underlying": pd.NA,
    })
    return df


def normalise(blob: bytes, d: date) -> pd.DataFrame:
    raw = _read_csv(blob)
    df = _normalise_udiff(raw) if "FinInstrmTp" in raw.columns else _normalise_legacy(raw)
    df["lot_size"] = pd.to_numeric(df["lot_size"], errors="coerce")
    df["underlying"] = pd.to_numeric(df["underlying"], errors="coerce")
    return df[COLUMNS].reset_index(drop=True)


def day_parquet(d: date) -> Path:
    return FO / f"{d:%Y}" / f"fo_{d:%Y%m%d}.parquet"


def build_day(d: date, symbols=DEFAULT_SYMBOLS, session=None, force=False) -> str:
    """Download + normalise one day. Returns 'ok' | 'cached' | 'holiday'."""
    out = day_parquet(d)
    if out.exists() and not force:
        return "cached"
    blob = fetch(d, session=session)
    if blob is None:
        return "holiday"
    df = normalise(blob, d)
    if symbols:
        df = df[df["symbol"].isin(list(symbols))]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="zstd")
    return "ok"


def build_range(start: date, end: date, symbols=DEFAULT_SYMBOLS, workers: int = 6,
                force: bool = False, progress=True) -> dict:
    """Walk every weekday in [start, end]; NSE 404s resolve the holiday calendar."""
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5]

    counts = {"ok": 0, "cached": 0, "holiday": 0, "error": 0}
    errors = []

    def work(d):
        sess = requests.Session()
        try:
            return d, build_day(d, symbols=symbols, session=sess, force=force), None
        except Exception as exc:
            return d, "error", exc

    it = range(0, len(days), 200)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk_start in it:
            chunk = days[chunk_start:chunk_start + 200]
            for d, status, exc in pool.map(work, chunk):
                counts[status] += 1
                if exc is not None:
                    errors.append((d, repr(exc)))
            if progress:
                done = min(chunk_start + 200, len(days))
                print(f"  bhavcopy {done}/{len(days)} days  {counts}", flush=True)

    return {"counts": counts, "errors": errors, "n_days": len(days)}


def load(start: date | None = None, end: date | None = None,
         symbol: str | None = None) -> pd.DataFrame:
    """Concatenate the normalised per-day parquet files."""
    files = sorted(FO.glob("*/fo_*.parquet"))
    if start or end:
        keep = []
        for f in files:
            d = date.fromisoformat(f.stem[3:7] + "-" + f.stem[7:9] + "-" + f.stem[9:11])
            if start and d < start:
                continue
            if end and d > end:
                continue
            keep.append(f)
        files = keep
    if not files:
        raise FileNotFoundError(f"no normalised bhavcopy under {FO}; run the download first")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if symbol:
        df = df[df["symbol"] == symbol]
    return df.sort_values(["date", "expiry", "strike", "opt_type"]).reset_index(drop=True)
