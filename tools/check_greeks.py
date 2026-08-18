"""Verify every greek against central finite differences of the pricer.

Getting this test right is harder than it looks, and both failure modes are
properties of the TEST, not of the pricer:

  noise      a central difference of a price of order 1e4 in double precision
             carries a floor of eps*price/h (and eps*price/h^2 for gamma). Deep
             ITM vega and gamma are ~1e-11, orders of magnitude below it, so a
             relative-error test there measures nothing but its own rounding.
  truncation a step that is large in absolute terms straddles the payoff kink
             for short-dated low-vol contracts, where the price is nearly
             max(F-K, 0). A fixed step is simply the wrong instrument.
  scale      a random cloud contains puts worth 1e-176 rupees (|d1| > 7). Their
             greeks are exactly 0 to double precision while the FD returns
             denormal garbage, so ANY relative test fails. Nothing about such a
             contract is measurable or matters, so they are excluded and counted.

So every step is scaled to the contract's own natural width, L = F*sigma*sqrt(T)
(one standard deviation of the forward over the option's life), and each greek
is judged against a tolerance built from its own noise floor.

Run: python tools/check_greeks.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from virain.black76 import price, greeks, implied_vol

EPS = np.finfo(float).eps
rng = np.random.default_rng(0)
N = 20000
F = rng.uniform(15000, 30000, N)
K = F * np.exp(rng.uniform(-0.35, 0.35, N))
T = rng.uniform(1 / 365, 2.0, N)
r = rng.uniform(0.0, 0.12, N)
s = rng.uniform(0.05, 1.2, N)
c = rng.random(N) < 0.5

# keep only economically meaningful contracts: at least a hundredth of a paisa
_px0 = price(F, K, T, r, s, c)
_keep = _px0 > 1e-4
_dropped = int((~_keep).sum())
F, K, T, r, s, c = (a[_keep] for a in (F, K, T, r, s, c))
N = len(F)

g = greeks(F, K, T, r, s, c)
px = price(F, K, T, r, s, c)
kw = dict(F=F, K=K, T=T, r=r, sigma=s, is_call=c)


L = F * s * np.sqrt(T)          # natural width of the forward distribution


def _c1(arg, h):
    up, dn = dict(kw), dict(kw)
    up[arg], dn[arg] = kw[arg] + h, kw[arg] - h
    return (price(**up) - price(**dn)) / (2 * h)


def _c2(arg, h):
    up, dn = dict(kw), dict(kw)
    up[arg], dn[arg] = kw[arg] + h, kw[arg] - h
    return (price(**up) - 2 * px + price(**dn)) / h ** 2


def d1_fd(arg, h):
    """Richardson-extrapolated first derivative: O(h^4) instead of O(h^2).

    Plain central differences carry a truncation error proportional to
    d3V/dx3, which explodes in the far tail (it scales as d1^4 near |d1| ~ 10).
    Extrapolating over h and h/2 cancels the leading term, so the far tail stops
    dominating the error budget.
    """
    return (4 * _c1(arg, h / 2) - _c1(arg, h)) / 3, 2 * EPS * np.abs(px) / h


def d2_fd(arg, h):
    return ((4 * _c2(arg, h / 2) - _c2(arg, h)) / 3,
            16 * EPS * np.abs(px) / h ** 2)


cases = []
n, fl = d1_fd("F", L * 1e-3);   cases.append(("delta_f", g["delta_f"], n, fl))
n, fl = d2_fd("F", L * 4e-2);   cases.append(("gamma_f", g["gamma_f"], n, fl))
n, fl = d1_fd("sigma", s * 1e-3); cases.append(("vega", g["vega"], n / 100, fl / 100))
n, fl = d1_fd("T", T * 1e-3);   cases.append(("theta", g["theta"], -n / 365, fl / 365))
n, fl = d1_fd("r", 1e-6);       cases.append(("rho", g["rho"], n / 100, fl / 100))

ok = True
print(f"greeks vs central finite differences, {N:,} random contracts")
print(f"  ({_dropped:,} of {N + _dropped:,} dropped: price < 1e-4, greeks are 0 to double precision)")
for name, analytic, numeric, floor in cases:
    tol = 1e-4 * np.abs(analytic) + 100 * floor
    bad = np.abs(analytic - numeric) > tol
    nbad = int(np.nansum(bad))
    ok &= nbad == 0
    resolved = np.abs(numeric) > 10 * floor
    rel = np.abs(analytic - numeric)[resolved] / np.abs(numeric)[resolved]
    print(f"  {name:<8} {nbad:>3} outside tolerance | "
          f"max rel err {np.nanmax(rel):.2e} over {resolved.sum():,} FD-resolvable "
          f"| {'PASS' if nbad == 0 else 'FAIL'}")

# solver round-trip, restricted to contracts whose price actually depends on vol
print(f"\nsolver round-trip")
iv, st = implied_vol(px, F, K, T, r, c)
live = (st == 0) & (np.abs(g["vega"]) > 1e-4)
err = np.abs(iv - s)[live]
good = np.nanmax(err) < 1e-9
ok &= good
print(f"  max |iv - true| {np.nanmax(err):.2e} over {live.sum():,} vol-sensitive "
      f"contracts | {'PASS' if good else 'FAIL'}")
dead = (st == 0) & (np.abs(g["vega"]) <= 1e-4)
print(f"  ({dead.sum():,} contracts excluded: vega below 1e-4, so sigma is not")
print(f"   identifiable from the price at double precision - not a solver defect)")

# put-call parity of the pricer itself
lhs = price(F, K, T, r, s, True) - price(F, K, T, r, s, False)
pe = np.nanmax(np.abs(lhs - np.exp(-r * T) * (F - K)))
good = pe < 1e-9
ok &= good
print(f"\nput-call parity of the pricer")
print(f"  max abs error {pe:.2e} | {'PASS' if good else 'FAIL'}")

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
