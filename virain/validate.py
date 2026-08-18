"""Independent checks on the built dataset.

Computed greeks are only as good as the forward and the price that went in, so
each check targets one link in that chain:

  forward   parity forward vs the traded future, on monthly expiries where both
            exist. These are derived by completely different routes; agreement
            to a few bp means the parity extraction is sound.
  vol level ATM 30-day IV vs India VIX. VIX is a variance-swap fair vol over the
            whole strip so it sits ABOVE ATM IV by the skew premium, but the two
            must track each other tightly. Low correlation means broken IV.
  solver    re-price at the solved IV and compare to the input price.
  rate      rebuild the WHOLE pipeline at 4% and 9% - forward extraction
            included - to measure how much r actually matters end to end.

    python -m virain.validate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bhavcopy, black76, chain, spot as spot_mod


def atm_iv_term(df: pd.DataFrame, target_dte: int = 30) -> pd.DataFrame:
    """Per-date ATM implied vol interpolated to `target_dte` in total variance."""
    liq = df[df["liquid"] & df["otm"] & (df["dte"] >= 1)]
    near = liq.assign(absm=liq["log_moneyness"].abs())
    near = near.sort_values(["date", "expiry", "absm"])
    atm = (near.groupby(["date", "expiry"])
               .head(4)
               .groupby(["date", "expiry"], as_index=False)
               .agg(iv=("iv", "mean"), dte=("dte", "first")))

    out = []
    for d, g in atm.groupby("date"):
        g = g.sort_values("dte")
        if len(g) < 2 or g["dte"].min() > target_dte or g["dte"].max() < target_dte:
            continue
        var = (g["iv"] ** 2 * g["dte"]).to_numpy()      # total variance
        v = np.interp(target_dte, g["dte"].to_numpy(), var)
        out.append((d, float(np.sqrt(v / target_dte))))
    return pd.DataFrame(out, columns=["date", "atm_iv"])


def report(symbol: str = "NIFTY") -> dict:
    df = chain.load(symbol)
    res = {}
    line = lambda s: print(s, flush=True)

    line("=" * 68)
    line(f"  virain validation report - {symbol}")
    line("=" * 68)

    # ---- coverage -----------------------------------------------------------
    days = df["date"].nunique()
    res["rows"], res["days"] = len(df), days
    line(f"\nCOVERAGE")
    line(f"  {len(df):,} contract-days over {days:,} sessions "
         f"({df.date.min().date()} -> {df.date.max().date()})")
    line(f"  liquid rows: {df['liquid'].sum():,} ({df['liquid'].mean():.1%})")
    line(f"  IV solved:   {df['iv'].notna().mean():.1%} of all rows, "
         f"{df.loc[df.contracts > 0, 'iv'].notna().mean():.1%} of traded rows")
    sc = {0: "ok", 1: "at/below intrinsic", 2: "above no-arb cap",
          3: "bad input/expired", 4: "solver hit bound"}
    for k, v in df["iv_status"].value_counts().items():
        line(f"    status {k} {sc.get(k,'?'):<20} {v:>10,} ({v/len(df):.2%})")
    line(f"  forward source: " + ", ".join(
        f"{k}={v:,}" for k, v in
        df.groupby(['date', 'expiry']).fwd_source.first().value_counts().items()))

    # ---- forward vs traded future ------------------------------------------
    fo = bhavcopy.load(symbol=symbol)
    fut = fo[(fo["opt_type"] == "FUT") & (fo["contracts"] > 0)][["date", "expiry", "close"]]
    fut = fut.rename(columns={"close": "fut_close"})
    par = (df[df["fwd_source"] == "parity"]
             .groupby(["date", "expiry"], as_index=False)
             .agg(forward=("forward", "first"), dte=("dte", "first")))
    m = par.merge(fut, on=["date", "expiry"], how="inner")
    m["err_bp"] = (m["forward"] / m["fut_close"] - 1) * 1e4
    res["fwd_err_bp_median"] = float(m["err_bp"].abs().median())
    line(f"\nFORWARD  parity vs traded future ({len(m):,} matched monthly expiries)")
    line(f"  |error| median {m.err_bp.abs().median():.2f} bp   "
         f"p90 {m.err_bp.abs().quantile(.9):.2f} bp   "
         f"p99 {m.err_bp.abs().quantile(.99):.2f} bp")
    line(f"  signed bias {m.err_bp.median():+.2f} bp")

    # ---- solver round-trip --------------------------------------------------
    s = df[df["liquid"]].sample(min(200_000, int(df["liquid"].sum())), random_state=0)
    rp = black76.price(s.forward, s.strike, s["T"], s.r, s.iv, (s.opt_type == "CE").to_numpy())
    err = np.abs(rp - s.price.to_numpy())
    res["reprice_max_err"] = float(np.nanmax(err))
    line(f"\nSOLVER  re-price at solved IV ({len(s):,} sampled liquid rows)")
    line(f"  max abs error {np.nanmax(err):.2e}   median {np.nanmedian(err):.2e}")

    # ---- ATM IV vs India VIX ------------------------------------------------
    atm = atm_iv_term(df, 30)
    vix = spot_mod.load("INDIAVIX")[["date", "close"]].rename(columns={"close": "vix"})
    j = atm.merge(vix, on="date", how="inner").dropna()
    j["atm_pct"] = j["atm_iv"] * 100
    corr = j["atm_pct"].corr(j["vix"])
    res["vix_corr"] = float(corr)
    res["vix_spread"] = float((j["vix"] - j["atm_pct"]).median())
    line(f"\nVOL LEVEL  ATM 30d IV vs India VIX ({len(j):,} overlapping sessions)")
    line(f"  correlation {corr:.4f}")
    line(f"  VIX - ATM_IV: median {(j.vix - j.atm_pct).median():+.2f} vol pts  "
         f"(positive is expected: VIX carries the skew premium)")
    line(f"  our ATM IV  : min {j.atm_pct.min():.1f}  median {j.atm_pct.median():.1f}  "
         f"max {j.atm_pct.max():.1f}")
    line(f"  India VIX   : min {j.vix.min():.1f}  median {j.vix.median():.1f}  "
         f"max {j.vix.max():.1f}")

    # ---- rate sensitivity ---------------------------------------------------
    # Must rebuild the FORWARD too, not just re-solve at a different r. Under
    # parity F = K + e^{rT}(C-P), so a rate change scales only the small (C-P)
    # term, and near the money that barely moves F at all. Bumping the whole
    # forward by e^{dr*T} instead would overstate the effect by ~500x.
    sample_dates = sorted(pd.Series(df["date"].unique()).sample(
        min(60, df["date"].nunique()), random_state=7))
    key = ["date", "expiry", "strike", "opt_type"]
    builds = {}
    for r_test in (0.04, 0.09):
        b = chain.build(symbol=symbol, rate_default=r_test, write=False,
                        progress=False, dates=sample_dates)
        builds[r_test] = b[b["liquid"]].set_index(key)[["iv", "forward"]]
    j2 = builds[0.04].join(builds[0.09], lsuffix="_lo", rsuffix="_hi", how="inner")
    d_iv = (j2["iv_hi"] - j2["iv_lo"]).abs() * 100
    d_fwd = (j2["forward_hi"] / j2["forward_lo"] - 1).abs() * 1e4
    res["rate_sens_vol_pts"] = float(d_iv.median())
    line(f"\nRATE SENSITIVITY  full rebuild at r = 4% vs 9% "
         f"({len(j2):,} liquid rows over {len(sample_dates)} sampled sessions)")
    line(f"  forward moves : median {d_fwd.median():.3f} bp   p95 {d_fwd.quantile(.95):.3f} bp")
    line(f"  |IV change|   : median {d_iv.median():.4f} vol pts   "
         f"p95 {d_iv.quantile(.95):.4f} vol pts")

    # ---- smile shape --------------------------------------------------------
    liq = df[df["liquid"] & df["otm"] & (df["dte"].between(20, 40))]
    bins = pd.cut(liq["log_moneyness"], [-.30, -.10, -.05, -.02, .02, .05, .10, .30])
    line(f"\nSMILE  median IV by log-moneyness, 20-40 dte, OTM side only")
    for b, v in liq.groupby(bins, observed=True)["iv"].median().items():
        line(f"  {str(b):>16}  {v*100:5.2f}%")
    line(f"  -> put wing above call wing is the equity index skew; if it were flat")
    line(f"     or inverted the forward would be wrong.")
    line("")
    return res


if __name__ == "__main__":
    report()
