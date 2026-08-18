"""End-to-end build: spot -> bhavcopy -> normalised F&O -> chain with IV + greeks.

    python -m virain.build --years 5
    python -m virain.build --start 2020-08-18 --end 2026-08-18 --symbols NIFTY
    python -m virain.build --skip-download          # re-run the modelling only
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

from . import bhavcopy, chain, spot
from .config import DEFAULT_RATE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="virain.build")
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat, default=date.today())
    ap.add_argument("--symbols", nargs="+", default=["NIFTY"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE,
                    help="fallback annualised rate; overridden by data/rates.csv")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-spot", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-parse cached zips")
    args = ap.parse_args(argv)

    end = args.end
    start = args.start or (end - timedelta(days=int(args.years * 365.25)))
    print(f"range {start} -> {end}   symbols={args.symbols}")

    if not args.skip_spot:
        print("[1/3] index history")
        for name in ("NIFTY", "INDIAVIX"):
            df = spot.download(name, start, end)
            print(f"  {name}: {len(df):,} sessions {df.date.min().date()} -> {df.date.max().date()}")

    if not args.skip_download:
        print("[2/3] F&O bhavcopy")
        res = bhavcopy.build_range(start, end, symbols=tuple(args.symbols),
                                   workers=args.workers, force=args.force)
        print(f"  {res['counts']}")
        if res["errors"]:
            print(f"  {len(res['errors'])} days failed, first few: {res['errors'][:5]}")

    print("[3/3] chain: forwards, implied vol, greeks")
    for sym in args.symbols:
        df = chain.build(symbol=sym, rate_default=args.rate)
        liq = df["liquid"].mean()
        print(f"  {sym}: {len(df):,} contract-days, {liq:.1%} pass the liquidity filter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
