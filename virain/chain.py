"""Assemble the option chain: forward extraction, implied vol, greeks.

The forward is the part everyone gets wrong. NIFTY options are European on the
index, but discounting them off spot forces you to invent a dividend yield.
Futures fix that - except NIFTY futures are monthly while options expire weekly,
so most expiries have no matching future at all.

So the forward is recovered from the chain itself, via put-call parity:

    C(K) - P(K) = e^{-rT} (F - K)

so F = K + e^{rT} (C(K) - P(K)) at every strike, and the market forward is the
robust centre of those per-strike estimates over the liquid near-ATM strikes.
No dividend yield is assumed and no futures contract is required, so weekly
expiries are handled exactly like monthlies.

The discount factor is taken from an external rate rather than fitted from the
parity slope: over a 2-day expiry e^{-rT} sits within 4bp of 1, far under the
0.05 tick, so the slope carries no usable rate information and fitting it
produces wild rates. The forward LEVEL, by contrast, is identified precisely.
This also makes r self-cancelling to first order - the same e^{rT} that lifts
the forward is divided back out when the option is priced.

Only genuinely TRADED strikes feed the regression. NSE's settlement prices for
illiquid strikes are model output, so including them would make the fit circular.

Fallbacks, in order, when a chain is too thin to fit:
    future      close of a same-expiry future, when one exists
    future_itp  log-basis interpolated across the monthly futures curve
    spot_carry  F = S e^{rT} at the fallback rate (last resort, flagged)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import bhavcopy, black76, rates as rates_mod, spot as spot_mod
from .config import CHAIN, DAYS_PER_YEAR, DEFAULT_RATE

_MIN_PAIRS = 3          # liquid strikes with BOTH legs traded
_MONEY_BAND = 0.10      # |K/S - 1| window the fit may draw from
_N_ATM = 6              # nearest-the-money pairs actually used
_FWD_LO, _FWD_HI = 0.90, 1.10


@dataclass
class Forward:
    fwd: float
    rate: float
    source: str
    n_pairs: int
    disp_bp: float = np.nan


def _fit_parity(strikes, cmid, pmid, T, r, spot_px) -> Forward | None:
    """F = median over near-ATM strikes of  K + e^{rT}(C - P).

    Median, not mean: one stale print on a single strike would otherwise drag
    the whole forward. disp_bp reports how much the per-strike estimates
    disagree, which is the honest data-quality signal for that chain.
    """
    if len(strikes) < _MIN_PAIRS or T <= 0:
        return None
    near = np.argsort(np.abs(strikes - spot_px))[:_N_ATM]
    est = strikes[near] + np.exp(r * T) * (cmid[near] - pmid[near])
    est = est[np.isfinite(est)]
    if len(est) < _MIN_PAIRS:
        return None
    fwd = float(np.median(est))
    if not (_FWD_LO <= fwd / spot_px <= _FWD_HI):
        return None
    disp = float((np.percentile(est, 90) - np.percentile(est, 10)) / fwd * 1e4)
    return Forward(fwd, float(r), "parity", int(len(est)), disp)


def _futures_curve(fut_day: pd.DataFrame, spot_px: float, asof) -> dict:
    """Map expiry -> (F, T) plus a log-basis interpolator over the monthly curve."""
    pts = {}
    for _, row in fut_day.iterrows():
        T = (row["expiry"] - asof).days / DAYS_PER_YEAR
        px = row["close"] if row["contracts"] > 0 else row["settle"]
        if T > 0 and px and px > 0:
            pts[row["expiry"]] = (float(px), T)
    return pts


def _forward_for(expiry, T, spot_px, fut_pts, rate_default) -> Forward:
    if expiry in fut_pts:
        return Forward(fut_pts[expiry][0], rate_default, "future", 0)
    if len(fut_pts) >= 2:
        Ts = np.array([t for _, t in fut_pts.values()])
        bs = np.array([np.log(f / spot_px) / t for f, t in fut_pts.values()])
        order = np.argsort(Ts)
        b = float(np.interp(T, Ts[order], bs[order]))   # flat outside the curve
        return Forward(float(spot_px * np.exp(b * T)), rate_default, "future_itp", 0)
    return Forward(float(spot_px * np.exp(rate_default * T)), rate_default, "spot_carry", 0)


def build(symbol: str = "NIFTY", rate_default: float = DEFAULT_RATE,
          keep_expiry_day: bool = False, write: bool = True,
          progress: bool = True, dates=None) -> pd.DataFrame:
    fo = bhavcopy.load(symbol=symbol)
    if dates is not None:
        fo = fo[fo["date"].isin(pd.to_datetime(pd.Index(dates)))]
    px_spot = spot_mod.load("NIFTY").set_index("date")["close"]

    opts = fo[fo["opt_type"].isin(["CE", "PE"])].copy()
    futs = fo[fo["opt_type"] == "FUT"].copy()

    # traded contracts print a real close; untraded ones carry a stale previous
    # close, so those fall back to NSE's settlement price.
    traded = opts["contracts"].fillna(0) > 0
    opts["price"] = np.where(traded, opts["close"], opts["settle"])
    opts["price_source"] = np.where(traded, "close", "settle")

    opts["T"] = (opts["expiry"] - opts["date"]).dt.days / DAYS_PER_YEAR
    opts["dte"] = (opts["expiry"] - opts["date"]).dt.days
    if not keep_expiry_day:
        opts = opts[opts["T"] > 0]

    # spot: prefer the index close, fall back to the bhavcopy underlying print
    opts["spot"] = opts["date"].map(px_spot)
    opts["spot"] = opts["spot"].fillna(opts["underlying"])
    opts = opts[opts["spot"].notna() & (opts["spot"] > 0)]

    uniq_dates = pd.Series(sorted(opts["date"].unique()))
    rate_by_date = dict(zip(uniq_dates, rates_mod.load(uniq_dates, rate_default)))

    fwd_rows = []
    fut_by_day = {d: g for d, g in futs.groupby("date")}
    days = list(opts.groupby("date"))
    for i, (d, day) in enumerate(days):
        spot_px = float(day["spot"].iloc[0])
        fut_pts = _futures_curve(fut_by_day.get(d, futs.iloc[:0]), spot_px, d)
        for expiry, grp in day.groupby("expiry"):
            T = float(grp["T"].iloc[0])
            liq = grp[(grp["contracts"].fillna(0) > 0) & (grp["price"] > 0)]
            wide = liq[(liq["strike"] / spot_px - 1).abs() <= _MONEY_BAND]
            ce = wide[wide["opt_type"] == "CE"].set_index("strike")["price"]
            pe = wide[wide["opt_type"] == "PE"].set_index("strike")["price"]
            common = ce.index.intersection(pe.index)
            r_d = float(rate_by_date.get(d, rate_default))
            fit = None
            if len(common) >= _MIN_PAIRS:
                fit = _fit_parity(common.values.astype(float),
                                  ce.loc[common].values.astype(float),
                                  pe.loc[common].values.astype(float), T, r_d, spot_px)
            if fit is None:
                fit = _forward_for(expiry, T, spot_px, fut_pts, r_d)
            fwd_rows.append((d, expiry, fit.fwd, fit.rate, fit.source,
                             fit.n_pairs, fit.disp_bp))
        if progress and i % 250 == 0:
            print(f"  forwards {i}/{len(days)} days", flush=True)

    fwd = pd.DataFrame(fwd_rows, columns=["date", "expiry", "forward", "r",
                                          "fwd_source", "n_pairs", "fwd_disp_bp"])
    df = opts.merge(fwd, on=["date", "expiry"], how="left")
    df = df[df["price"].notna() & (df["price"] > 0)]

    is_call = (df["opt_type"] == "CE").to_numpy()
    F = df["forward"].to_numpy(float)
    K = df["strike"].to_numpy(float)
    T = df["T"].to_numpy(float)
    r = df["r"].to_numpy(float)
    P = df["price"].to_numpy(float)

    if progress:
        print(f"  solving implied vol for {len(df):,} contracts", flush=True)
    iv, status = black76.implied_vol(P, F, K, T, r, is_call)
    df["iv"] = iv
    df["iv_status"] = status

    g = black76.greeks(F, K, T, r, iv, is_call, S=df["spot"].to_numpy(float))
    for k in ("delta_f", "delta_s", "gamma_f", "gamma_s", "vega", "theta", "rho",
              "vanna", "vomma", "d1", "d2"):
        df[k] = g[k]
    df["log_moneyness"] = np.log(K / F)

    # The recommended research filter, precomputed. NSE assigns theoretical
    # settlement prices to contracts that never traded; inverting those gives a
    # number, not a market vol. Anything failing `liquid` is model output
    # dressed as data - keep it in the file, but opt in deliberately.
    df["liquid"] = (
        (df["contracts"].fillna(0) > 0)
        & (df["oi"].fillna(0) > 0)
        & (df["price"] >= 0.05)
        & (df["iv"].notna())
        & (df["fwd_source"] == "parity")
        & (df["dte"] <= 400)
    )

    # Out-of-the-money side. A deep ITM option is almost all intrinsic value, so
    # its tiny time value - and hence its implied vol - is dominated by the
    # bid-ask spread: NIFTY deep ITM puts routinely invert to 50-90% vol that no
    # one would ever trade. Every smile, skew or vol-surface study should use the
    # OTM side only; put-call parity means you lose no information by doing so.
    df["otm"] = np.where(is_call, K >= F, K <= F)

    keep = ["date", "symbol", "expiry", "dte", "T", "strike", "opt_type",
            "price", "price_source", "close", "settle", "contracts", "oi", "chg_oi",
            "spot", "forward", "fwd_source", "n_pairs", "fwd_disp_bp", "r",
            "iv", "iv_status", "log_moneyness",
            "delta_f", "delta_s", "gamma_f", "gamma_s", "vega", "theta", "rho",
            "vanna", "vomma", "d1", "d2", "liquid", "otm"]
    df = df[keep].sort_values(["date", "expiry", "strike", "opt_type"]).reset_index(drop=True)

    if write:
        CHAIN.mkdir(parents=True, exist_ok=True)
        for year, part in df.groupby(df["date"].dt.year):
            out = CHAIN / f"{symbol.lower()}_options_{year}.parquet"
            part.to_parquet(out, index=False, compression="zstd")
            if progress:
                print(f"  wrote {out}  ({len(part):,} rows)", flush=True)
    return df


def load(symbol: str = "NIFTY") -> pd.DataFrame:
    files = sorted(CHAIN.glob(f"{symbol.lower()}_options_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no chain files in {CHAIN}; run the build first")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
