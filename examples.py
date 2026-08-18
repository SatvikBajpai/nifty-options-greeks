"""Worked examples against the built dataset.

    python examples.py
"""
import numpy as np
import pandas as pd

from virain import chain, spot

pd.set_option("display.width", 120)

df = chain.load("NIFTY")
liq = df[df["liquid"]]
print(f"loaded {len(df):,} contract-days, {len(liq):,} liquid\n")

# 1. the original ask: NIFTY 50 daily closes
nifty = spot.load("NIFTY")
print("1. NIFTY 50 daily close")
print(f"   {len(nifty):,} sessions, {nifty.date.min().date()} -> {nifty.date.max().date()}")
print(nifty.tail(3).to_string(index=False), "\n")

# 2. ATM implied vol history at ~30 dte
from virain.validate import atm_iv_term
atm = atm_iv_term(df, 30)
print("2. ATM 30-day implied vol")
print(f"   {len(atm):,} sessions | median {atm.atm_iv.median()*100:.2f}%  "
      f"min {atm.atm_iv.min()*100:.2f}%  max {atm.atm_iv.max()*100:.2f}%")
print(atm.tail(3).to_string(index=False), "\n")

# 3. the smile on the most recent session
last = liq[liq["date"] == liq["date"].max()]
near = last[last["expiry"] == last["expiry"].min()]
print(f"3. smile on {last.date.max().date()}, expiry {near.expiry.min().date()} (OTM side)")
sm = (near[(near["log_moneyness"].abs() < 0.06) & near["otm"]]
      .sort_values("strike")[["strike", "opt_type", "price", "iv", "delta_s", "vega", "theta"]])
print(sm.to_string(index=False, float_format=lambda x: f"{x:,.4f}"), "\n")

# 4. 25-delta risk reversal: the standard skew measure
d = liq[liq["dte"].between(20, 40) & liq["otm"]].copy()
d["absdelta"] = d["delta_s"].abs()
puts = d[d.opt_type == "PE"].assign(dist=lambda x: (x.absdelta - 0.25).abs())
calls = d[d.opt_type == "CE"].assign(dist=lambda x: (x.absdelta - 0.25).abs())
p25 = puts.sort_values(["date", "dist"]).groupby("date").first()["iv"]
c25 = calls.sort_values(["date", "dist"]).groupby("date").first()["iv"]
rr = ((c25 - p25) * 100).dropna()
print("4. 25-delta risk reversal (call IV - put IV), 20-40 dte")
print(f"   {len(rr):,} sessions | median {rr.median():+.2f} vol pts  "
      f"(negative = puts bid, the normal equity skew)\n")

# 5. portfolio greeks of a short strangle, aggregated properly
print("5. short 25-delta strangle, greeks per lot on the last session")
strad = pd.concat([
    calls[calls.date == calls.date.max()].sort_values("dist").head(1),
    puts[puts.date == puts.date.max()].sort_values("dist").head(1),
])
if len(strad) == 2:
    lot = 75
    tot = (strad[["delta_s", "gamma_s", "vega", "theta"]] * -lot).sum()
    print(strad[["strike", "opt_type", "price", "iv", "delta_s", "vega", "theta"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"   net per lot: delta {tot.delta_s:+.2f}  gamma {tot.gamma_s:+.4f}  "
          f"vega {tot.vega:+.1f}/vol pt  theta {tot.theta:+.1f}/day")
