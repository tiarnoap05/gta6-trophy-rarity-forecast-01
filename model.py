"""Hierarchical saturating-growth forecaster for GTA 6 trophy earn rates.

The idea, in one paragraph: a trophy's earn rate on Exophase is the share of
tracked players who have earned it, and it grows like a saturating curve as
players play through the game -- fast for early chapter trophies, slow for the
platinum and 100% completion, and never above the share of players who will
ever reach it. So per trophy we fit r(t) = A - B*exp(-k*t), hours since
launch, by a grid search over k with least squares for A and B (B >= 0, so
the curve never falls, and A >= the last observed rate, so the asymptote sits
above where the trophy is now). The predicted growth to the target is then
shrunk toward the tier's median growth, with the weight on the trophy's own
fit given by how much of its own movement the fit explains (R^2); a trophy
that never moved borrows its tier's growth, and when nothing moved at all --
as in the practice cases, where no trophy moved -- every growth is zero and
the forecast degrades exactly to the persistence starter, which is the right
thing to do with no signal. The spread is honest: the starter's random-walk
term growing with the horizon, the fit's residual converted to logit units
and widened by how far the target is extrapolated, the uncertainty of
borrowing the tier's growth, and a floor for the curve form itself being
wrong (notably dilution: new tracked players who have not played yet drag
rates down, which no per-trophy curve sees).

Standard library only, deterministic, no network.
"""

import math
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

SIGMA_RW = 0.5        # logit random-walk sigma per sqrt(day), as the starter
FLOOR_SPREAD = 0.15   # logit, floor for the curve form being wrong
# Damp the fitted growth toward persistence: the fitted curve extrapolates
# 4.5 days from ~2.5 days of history, and a saturating fit that has not seen
# its bend yet over-promises. 0.55 was tuned on synthetic backtests under
# three ground-truth shapes (cohort, logit-linear, piecewise): it keeps most
# of the win where growth is real and nearly erases the loss where the truth
# flattens early.
GROWTH_DAMP = 0.55
MAX_SPREAD = 2.0      # logit, cap on the total spread
MIN_SPREAD = 0.15     # logit, floor on the total spread

# Grid of speeds k (per hour) for the saturating fit r(t) = A - B*exp(-k*t).
K_GRID = [0.002 * (1.35 ** i) for i in range(30)]  # 0.002 .. ~0.55


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def _parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _launch_time(case):
    """Hours are counted from the game's release date at 00:00 UTC."""
    try:
        rel = case.get("game", {}).get("release_date", "")
        dt = datetime.fromisoformat(rel)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        hist = case.get("history") or []
        if hist:
            return _parse_time(hist[0]["at"])
        return _parse_time(case["at"])


def _median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    if n % 2:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


def _fit_saturating(ts, rs):
    """Fit r(t) = A - B*exp(-k*t) by grid search over k, LS for A and B.

    Returns (growth_to_target_fn, resid_std, r_squared) where the function
    takes a target time and returns the fitted rate there, or None when the
    series carries no usable movement (fewer than 3 points, or a range below
    the page's 2-decimal rounding noise).
    """
    n = len(ts)
    if n < 3:
        return None
    r_lo, r_hi = min(rs), max(rs)
    if r_hi - r_lo < 0.02:
        return None
    r_mean = sum(rs) / n
    var = sum((r - r_mean) ** 2 for r in rs)
    if var <= 1e-12:
        return None
    best = None
    for k in K_GRID:
        gs = [math.exp(-k * t) for t in ts]
        # Least squares of r on [1, -g]: r = A - B*g.
        sg = sum(gs)
        sgg = sum(g * g for g in gs)
        sr = sum(rs)
        sgr = sum(g * r for g, r in zip(gs, rs))
        det = n * sgg - sg * sg
        if abs(det) < 1e-12:
            continue
        a = (sr * sgg - sg * sgr) / det
        b = (sg * sr - n * sgr) / det
        if b < 0:
            b = 0.0
            a = r_mean
        if a < r_hi:
            a = r_hi
        if a > 99.99:
            a = 99.99
        sse = sum((r - (a - b * g)) ** 2 for g, r in zip(gs, rs))
        if best is None or sse < best[0]:
            best = (sse, a, b, k)
    if best is None:
        return None
    sse, a, b, k = best
    resid = math.sqrt(sse / max(n - 2, 1))
    r2 = max(0.0, min(1.0, 1.0 - sse / var))
    return (a, b, k, resid, r2)


def _dlogit_dr(rate_pct):
    """d(logit)/d(rate in points): converts a rate-space std to logit units."""
    r = clamp(rate_pct)
    return 100.0 / (r * (100.0 - r))


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the expected five")
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    launch = _launch_time(case)
    t_target = (_parse_time(case["target"]) - launch).total_seconds() / 3600.0
    horizon_days = case["horizon_minutes"] / 1440.0

    # Per-trophy series: (hours since launch, rate in percent).
    series = []
    for i in range(len(trophies)):
        ts, rs = [], []
        for read in history:
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            t = (_parse_time(read["at"]) - launch).total_seconds() / 3600.0
            ts.append(max(t, 0.0))
            rs.append(rates[i])
        series.append((ts, rs))

    fits = []
    for ts, rs in series:
        fits.append(_fit_saturating(ts, rs))

    # Tier median predicted growth (only from trophies with a fit); a tier
    # with no fitted growth gets 0, i.e. persistence for its trophies, never
    # another tier's pace.
    growths = []
    for idx, (ts, rs) in enumerate(series):
        fit = fits[idx]
        if fit is None:
            growths.append(None)
            continue
        a, b, k, _, _ = fit
        r_last = rs[-1]
        r_pred = a - b * math.exp(-k * max(t_target, 0.0))
        growths.append(max(0.0, min(99.99 - r_last, r_pred - r_last)))
    tier_growth = {"platinum": 0.0, "gold": 0.0, "silver": 0.0, "bronze": 0.0}
    by_tier = {}
    for idx, trophy in enumerate(trophies):
        if growths[idx] is not None:
            by_tier.setdefault(trophy["tier"], []).append(growths[idx])
    for tier, gs in by_tier.items():
        tier_growth[tier] = _median(gs)

    rows = []
    for idx, trophy in enumerate(trophies):
        ts, rs = series[idx]
        fit = fits[idx]
        if not rs:
            # No observation at all: median 1%, very wide spread.
            med_logit = logit(1.0)
            spread = MAX_SPREAD
            rows.append({"id": trophy["id"],
                         "q": [round4(inv_logit(med_logit + z * spread)) for z in Z]})
            continue
        r_last = rs[-1]
        t_last = ts[-1]
        t_first = ts[0]
        span = max(t_last - t_first, 1.0)
        extrap = max(t_target - t_last, 0.0)

        tier_g = tier_growth.get(trophy["tier"], 0.0)
        if fit is not None:
            a, b, k, resid, r2 = fit
            lam = r2  # weight on the trophy's own fit: how much it explains
            g = lam * (growths[idx] or 0.0) + (1.0 - lam) * tier_g
            # Residual (rate points) -> logit, widened by extrapolation.
            fit_term = resid * _dlogit_dr(r_last + g) * (1.0 + 0.5 * extrap / span)
            tier_term = (1.0 - lam) * abs(tier_g) * _dlogit_dr(r_last + g)
        else:
            g = tier_g
            lam = 0.0
            fit_term = 0.0
            tier_term = abs(tier_g) * _dlogit_dr(r_last + g)

        r_med = min(99.99, max(0.01, r_last + GROWTH_DAMP * g))
        med_logit = logit(r_med)
        rw_var = (SIGMA_RW * math.sqrt(max(horizon_days, 0.0))) ** 2
        var = rw_var + fit_term * fit_term + tier_term * tier_term + FLOOR_SPREAD ** 2
        spread = min(MAX_SPREAD, max(MIN_SPREAD, math.sqrt(var)))
        rows.append({"id": trophy["id"],
                     "q": [round4(inv_logit(med_logit + z * spread)) for z in Z]})
    return {"trophies": rows}
