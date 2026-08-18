# virain

Five-plus years of NIFTY 50 daily index history and NSE F&O option prices, with
implied volatility and a full greek set computed from scratch under Black-76.

Everything here comes from free, official sources. No vendor, no API key.

## Why compute the greeks yourself

No free source publishes historical Greeks or IV for NSE options, and the paid
ones publish greeks computed under assumptions you cannot inspect - which is
strictly worse for research than greeks whose conventions you chose. NSE's
bhavcopy is the same raw feed the vendors resell. The only thing you buy from
them is the plumbing, and the plumbing is this repo.

## Quick start

```bash
pip install -r requirements.txt
python -m virain.build --years 5          # spot + bhavcopy + IV + greeks
python -m virain.validate                 # prove the numbers are sane
```

Roughly 6 minutes and ~1 GB on disk for six years of NIFTY. Re-runs are
incremental: raw zips are cached, so widening the symbol list or changing the
model costs a re-parse, never a re-download.

```python
from virain import chain, spot

nifty = spot.load("NIFTY")                # daily OHLC, what you originally asked for
df    = chain.load("NIFTY")               # option chain with IV + greeks
df    = df[df.liquid]                     # <- read the liquidity section below
```

## What the pipeline does

| step | module | output |
|---|---|---|
| NIFTY 50 + India VIX daily OHLC | `spot.py` | `data/spot/*.parquet`, `.csv` |
| F&O bhavcopy download, both schemas | `bhavcopy.py` | `data/raw/fo/YYYY/*.zip` |
| normalise onto one schema | `bhavcopy.py` | `data/fo/YYYY/*.parquet` |
| forward, implied vol, greeks | `chain.py` | `data/chain/nifty_options_YYYY.parquet` |

### Two bhavcopy schemas

NSE replaced the F&O bhavcopy format with UDiFF partway through the window.
The boundary is clean, with no gap and no overlap to reconcile:

- `<= 2023-12-29` legacy `/content/historical/DERIVATIVES/2023/AUG/fo14AUG2023bhav.csv.zip`
- `>= 2024-01-01` UDiFF `/content/fo/BhavCopy_NSE_FO_0_0_0_20250814_F_0000.csv.zip`

Both are parsed into one schema. Note the legacy files carry neither lot size
nor the underlying print, so those columns are null before 2024.

Market holidays are discovered rather than hardcoded: every weekday is requested,
a 404 means no session, and the 404 is memoised as a `.missing` marker so reruns
never re-ask.

## The forward is the whole game

NIFTY options are European on the index. Discounting them off spot forces you to
invent a dividend yield, and every error in that guess lands directly in your IV.
The usual advice is to use the same-expiry future instead - but **NIFTY futures
are monthly while options expire weekly**, so most expiries in this dataset have
no matching future at all.

So the forward is recovered from the option chain itself, by put-call parity:

```
C(K) - P(K) = e^{-rT} (F - K)      =>      F = K + e^{rT} (C(K) - P(K))
```

evaluated at each of the six near-the-money strikes where **both legs actually
traded**, then taken as a median for robustness against a single stale print.
No dividend assumption, no futures contract needed, weekly expiries handled
exactly like monthlies. This is the standard CBOE / OptionMetrics construction.

Only genuinely traded strikes feed this. NSE's settlement prices for illiquid
strikes are *model output*, so letting them set the forward would be circular.

Validation, on expiries where a traded future also exists (two completely
independent routes to the same number):

```
FORWARD  parity vs traded future (4,356 matched monthly expiries)
  |error| median 1.71 bp   p90 5.34 bp   p99 12.65 bp
  signed bias +0.58 bp
```

When a chain is too thin to fit, the build falls back and says so in
`fwd_source`: `parity` -> `future` -> `future_itp` (log-basis interpolated
across the monthly futures curve) -> `spot_carry`. In practice parity covers
everything inside 90 dte; the fallbacks only appear on long-dated contracts that
barely trade.

## The risk-free rate matters much less than you think

`r` defaults to a flat 6.5%. Drop a `data/rates.csv` with `date,rate` columns
(91-day T-bill or MIBOR, decimal or percent) to use a real curve.

Do not spend a weekend on this. The rate is *not fitted* from the parity slope
on purpose: over a 2-day expiry `e^{-rT}` sits within 4 bp of 1, far under the
0.05 tick, so the slope carries no usable rate information - trying to fit it
yields implied rates ranging from -6% to +51%. The forward *level*, by contrast,
is pinned down precisely.

Better still, r is largely self-cancelling. Under parity `F = K + e^{rT}(C-P)`,
so the rate scales only the *small* `(C-P)` term - and near the money `C-P` is
close to zero, so the forward hardly moves at all. Rebuilding the entire
pipeline (forward extraction included) at 4% and again at 9%:

```
RATE SENSITIVITY  full rebuild at r = 4% vs 9%
  forward moves : median 0.087 bp    p95 1.791 bp
  |IV change|   : median 0.0294 vol pts   p95 0.7425 vol pts
```

A **5 percentage point** rate error moves the median contract by three
hundredths of a vol point. Use the flat default and spend the time elsewhere.

The p95 of 0.74 vol points is the honest caveat: low-vega wings have little
price sensitivity to vol, so any input error gets amplified when inverted. If
your study lives in the far wings rather than near the money, supply a real
rate curve. For ATM and near-ATM work the flat default is fine.

Note this is easy to measure wrongly. Bumping the stored forward by `e^{dr*T}`
and re-solving - rather than re-extracting the forward - overstates the effect
by roughly 35x, because it applies the rate to the whole forward instead of to
`C-P`. `validate.py` rebuilds end to end for this reason.

## Which price is used

Untraded contracts do not print a meaningful close - in both schemas a contract
with zero volume carries the *previous* session's close, which can be stale by
hundreds of points. So:

```
price = close   if contracts > 0     (a real trade)
        settle  otherwise            (NSE's theoretical mark)
```

recorded per row in `price_source`.

## Liquidity: the filter that separates data from fiction

**This is the single most important thing in the repo.** NSE assigns theoretical
settlement prices to every listed contract, including strikes that have never
traded in their life. Inverting those through a solver returns a number, and
that number is not a market volatility - it is NSE's model, refracted through
yours. Feeding them into a study produces confident nonsense.

Every row is kept in the file, but the recommended filter is precomputed as
`liquid`:

```python
liquid = (contracts > 0) & (oi > 0) & (price >= 0.05)
         & iv.notna() & (fwd_source == "parity") & (dte <= 400)
```

Opt out deliberately, not accidentally.

### And use the OTM side for anything vol-related

A second flag, `otm`, marks calls above the forward and puts below it. A deep
ITM option is almost entirely intrinsic value, so its sliver of time value - and
therefore its implied vol - is dominated by the bid-ask spread. In this dataset
deep ITM NIFTY puts routinely invert to 50-90% vol that nobody would trade at.

Put-call parity means the OTM side carries the complete information, so you lose
nothing:

```python
smile = df[df.liquid & df.otm]      # 26.2% of all rows
```

Use `liquid & otm` for any smile, skew, or vol-surface work. `liquid` alone is
right when you want the traded universe (open interest studies, volume, flow).

## Greek conventions

Black-76 on the forward. Stated explicitly because this is exactly where two
datasets silently disagree:

| column | meaning |
|---|---|
| `delta_f` | dV/dF, forward delta |
| `delta_s` | dV/dS, spot delta = `delta_f * F/S` |
| `gamma_f`, `gamma_s` | second order in F and in S |
| `vega` | per **1 volatility point** (dV/dsigma / 100) |
| `theta` | per **calendar day** (dV/dt / 365) |
| `rho` | per **1% change in r** |
| `vanna`, `vomma` | per vol point, and per vol point squared |
| `iv` | annualised, decimal (0.14 = 14%) |
| `T` | calendar days / 365 |
| `liquid`, `otm` | the two research filters described above |
| `fwd_source`, `n_pairs`, `fwd_disp_bp` | how the forward was obtained, and how much the per-strike parity estimates disagreed |

Greeks are verified against central finite differences of the pricer to 6+
significant figures (`tools/check_greeks.py`).

`iv_status` explains every unsolved row: `0` ok, `1` price at or below
intrinsic, `2` above the no-arbitrage cap, `3` bad input or expired, `4` solver
hit a bound. Nothing fails silently.

Implied vol is solved by vectorised bisection, not Newton: vega collapses for
deep OTM strikes where Newton diverges quietly. 64 halvings of [1e-4, 5] resolve
sigma far below any meaningful precision, over the whole dataset in seconds.

Expiry-day rows (`T = 0`) are dropped by default - the data is end-of-day and
settlement is 15:30, so there is no time value left to invert. Pass
`keep_expiry_day=True` to retain them.

## Validation

`python -m virain.validate` checks each link in the chain independently:

- **forward** parity forward vs the traded future on monthly expiries
- **vol level** ATM 30-day IV vs India VIX (VIX sits above ATM IV by the skew
  premium; the two must still track tightly)
- **solver** re-price at the solved IV, compare to the input
- **rate** rebuild IV at 4% and 9% to measure how much r really matters
- **smile** median IV by log-moneyness; the put wing must sit above the call
  wing, because that is the equity index skew. Flat or inverted means the
  forward is wrong.

## Extending to other underlyings

`--symbols NIFTY BANKNIFTY` re-parses the cached zips - no new downloads. The
whole F&O universe is already in `data/raw/fo/`; the symbol list only controls
what gets normalised.

## Limits

- **End of day only.** Intraday history cannot be reconstructed from public
  sources after the fact. If you need minute bars or gamma-sensitive work, that
  is the one case worth paying a vendor for - buy three months, archive it,
  cancel.
- Legacy-format days (pre-2024) have no lot size or underlying print.
- Long-dated contracts (> 400 dte) are excluded by `liquid`; they barely trade
  and their forwards are extrapolated.
- Index data comes from Yahoo (`^NSEI`). Swap in NSE's own CSV if you need the
  official print to the last paisa.

## Licensing and redistribution

**The code is MIT. The data is not mine to give away, and probably not yours
either.** That distinction is the whole of this section.

This repository deliberately ships **no market data**. `.gitignore` excludes
`data/`, so cloning gets you the pipeline and you regenerate the dataset
yourself in about six minutes. That is not an oversight - it is the only
arrangement that is clearly permitted.

### Why the data stays out

NSE's copyright policy is explicit that the exchange owns the content and that
reproducing it elsewhere is not allowed:

> NSE is the owner of copyright in all Content featured on its Website/Mobile
> Application and no portion of the Content may be reproduced on or transmitted
> to or stored in any other website or in other form of electronic retrieval
> system or in any other form or by any other means.

with a carve-out that covers exactly what this pipeline does locally:

> However, users may view, print copies and download the Content for personal,
> non-commercial or educational purpose without in any way amending, altering,
> deleting or modifying any part of the Content and provide full acknowledgement
> that the Content originated from the Website/Mobile Application.

Separately, NSE's Data Sharing & Usage Policy states that NSE retains ownership
of all data and that subscribers may not redistribute Market Data except under
an agreement with NSE. Commercial redistribution is licensed through NSE Data &
Analytics Limited, a wholly owned subsidiary that exists for that purpose.

So: **downloading it yourself is explicitly fine. Publishing a copy is
explicitly the thing the policy prohibits.** A public GitHub repository is
"any other website ... or other form of electronic retrieval system".

### The derived data is not a loophole

It is tempting to think implied vols and greeks are your own computed facts and
therefore freely publishable. Be careful: `data/chain/*.parquet` also carries
`close`, `settle`, `contracts` and `oi` verbatim from the bhavcopy. Those files
are a reproduction of NSE's data with extra columns, not a clean derivative.

There is a real argument that raw prices are uncopyrightable facts - Indian
copyright law moved away from "sweat of the brow" in *Eastern Book Company v.
D.B. Modak* (2008), which requires a modicum of creativity - and India has no
EU-style sui generis database right. But that is a defence you would raise after
being asked to take something down, not a permission you have in advance, and
NSE's terms operate as contract independently of copyright.

### Other constraints worth knowing

- **Yahoo.** `spot.py` pulls `^NSEI` through yfinance, which scrapes Yahoo
  Finance. Yahoo's terms prohibit redistribution too, so the spot CSVs are no
  safer to publish than the bhavcopy.
- **Trademark.** NIFTY 50 is a registered trademark of NSE Indices Limited.
  Using it descriptively - "a pipeline for NIFTY 50 options data" - is ordinary
  nominative use. Do not use NSE branding or logos, do not name a project so it
  reads as official, and do not imply endorsement or affiliation.
- **Plenty of repos host bhavcopy archives anyway.** They mostly survive. That
  is a statement about how actively NSE enforces, not about whether it is
  permitted. Treat it as risk tolerance, not as precedent.

### If you want to publish data anyway

Ask NSE. Data licensing runs through NSE Data & Analytics Limited, and a
research or educational use is a conversation worth having rather than a
question to answer by assumption.

*None of this is legal advice - I am an engineer, not your lawyer. If real money
or a real business depends on the answer, ask one.*
