"""Black-76 pricing, a vectorised implied-vol solver, and greeks.

Black-76 prices an option on a *forward/futures* price F, so dividend yield and
cost of carry never appear explicitly - they are already inside F. That is the
right model for NIFTY options once you recover F from the option chain itself
(see chain.py), rather than plugging in spot and guessing a dividend yield.

Conventions of the returned greeks:
    delta_f   dV/dF   (forward delta, per 1 point of the forward)
    delta_s   dV/dS   (spot delta; = delta_f * F/S)
    gamma_f   d2V/dF2 (per 1 point of the forward, squared)
    vega      per 1 volatility POINT (i.e. dV/dsigma / 100)
    theta     per CALENDAR day (dV/dt / 365)
    rho       per 1% change in r (dV/dr / 100)
    vanna     d2V/dFdsigma, per 1 vol point
    vomma     d2V/dsigma2, per 1 vol point squared
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _pdf(x):
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def price(F, K, T, r, sigma, is_call):
    """Black-76 price. All arguments broadcast; sigma/T may be zero."""
    F, K, T, r, sigma = np.broadcast_arrays(*[np.asarray(a, float) for a in (F, K, T, r, sigma)])
    is_call = np.asarray(is_call, bool)
    df = np.exp(-r * T)
    v = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * v * v) / v
        d2 = d1 - v
    call = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    out = np.where(is_call, call, put)
    # degenerate limit: no time value left
    intrinsic = df * np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    return np.where(v > 0, out, intrinsic)


def implied_vol(target, F, K, T, r, is_call, lo=1e-4, hi=5.0, iters=64):
    """Vectorised bisection implied vol.

    Bisection rather than Newton because option vega collapses for deep OTM
    strikes, where Newton diverges silently. 64 halvings of [1e-4, 5] resolve
    sigma to ~3e-16, so the iteration count is never the binding error.

    Returns (sigma, status) where status is an int8 code:
        0 ok, 1 price at/below intrinsic, 2 price above the no-arb cap,
        3 non-positive price or expired, 4 solver hit a bound
    """
    target, F, K, T, r = np.broadcast_arrays(*[np.asarray(a, float) for a in (target, F, K, T, r)])
    is_call = np.broadcast_to(np.asarray(is_call, bool), target.shape)

    df = np.exp(-r * T)
    intrinsic = df * np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    cap = df * np.where(is_call, F, K)          # C <= df*F, P <= df*K

    status = np.zeros(target.shape, dtype=np.int8)
    bad = ~np.isfinite(target) | (target <= 0) | ~np.isfinite(F) | (T <= 0) | (K <= 0)
    status = np.where(bad, 3, status)
    status = np.where(~bad & (target <= intrinsic + 1e-10), 1, status)
    status = np.where(~bad & (target >= cap - 1e-10), 2, status)

    live = status == 0
    sigma = np.full(target.shape, np.nan)
    if not live.any():
        return sigma, status

    a = np.full(live.sum(), lo)
    b = np.full(live.sum(), hi)
    Fl, Kl, Tl, rl, tl, cl = (F[live], K[live], T[live], r[live], target[live], is_call[live])
    for _ in range(iters):
        m = 0.5 * (a + b)
        too_low = price(Fl, Kl, Tl, rl, m, cl) < tl
        a = np.where(too_low, m, a)
        b = np.where(too_low, b, m)
    m = 0.5 * (a + b)

    sigma[live] = m
    at_bound = (m <= lo * 1.01) | (m >= hi * 0.99)
    idx = np.flatnonzero(live)
    status[idx[at_bound]] = 4
    sigma[idx[at_bound]] = np.nan
    return sigma, status


def greeks(F, K, T, r, sigma, is_call, S=None):
    """Full greek set for Black-76. Returns a dict of arrays."""
    F, K, T, r, sigma = np.broadcast_arrays(*[np.asarray(a, float) for a in (F, K, T, r, sigma)])
    is_call = np.broadcast_to(np.asarray(is_call, bool), F.shape)

    df = np.exp(-r * T)
    sqrtT = np.sqrt(T)
    v = sigma * sqrtT
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * v * v) / v
        d2 = d1 - v
        nd1 = _pdf(d1)

        Nd1, Nd2 = norm.cdf(d1), norm.cdf(d2)
        delta_f = df * np.where(is_call, Nd1, Nd1 - 1.0)
        gamma_f = df * nd1 / (F * v)
        vega = F * df * nd1 * sqrtT

        carry = np.where(is_call, r * (F * df * Nd1 - K * df * Nd2),
                         r * (K * df * (1 - Nd2) - F * df * (1 - Nd1)))
        theta = -F * df * nd1 * sigma / (2.0 * sqrtT) + carry

        vanna = -df * nd1 * d2 / sigma
        vomma = vega * d1 * d2 / sigma
        px = price(F, K, T, r, sigma, is_call)
        rho = -T * px

    out = {
        "delta_f": delta_f,
        "gamma_f": gamma_f,
        "vega": vega / 100.0,
        "theta": theta / 365.0,
        "rho": rho / 100.0,
        "vanna": vanna / 100.0,
        "vomma": vomma / 10000.0,
        "d1": d1,
        "d2": d2,
        "model_price": px,
    }
    if S is not None:
        S = np.asarray(S, float)
        out["delta_s"] = delta_f * (F / S)
        out["gamma_s"] = gamma_f * (F / S) ** 2
    return out
