[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tiarnoap05/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tiarnoap05/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA 6 trophy rarity forecast

A program that forecasts, for every PlayStation 5 trophy of Grand Theft Auto
VI, the percent of players Exophase tracks who will have earned it a week
after launch (2026-11-26T00:00:00Z), as five quantiles per trophy, scored by
pinball loss on the logit scale.

## The model

A trophy's earn rate grows like a saturating curve as players play through
the game, so per trophy we fit `r(t) = A - B*exp(-k*t)` (hours since launch)
by a grid search over `k` with least squares for `A` and `B`. The predicted
growth to the target is shrunk toward the tier's median growth, weighted by
how much of the trophy's own movement the fit explains; a trophy that never
moved borrows its tier's growth, and when nothing moved at all the forecast
degrades exactly to the persistence starter. The fitted growth is damped by
0.55: a saturating fit that has not seen its bend yet over-promises over a
4.5-day extrapolation. The spread is honest -- the starter's random-walk term
growing with the horizon, the fit's residual in logit units widened by how
far the target is extrapolated, the uncertainty of borrowing the tier's
growth, and a floor for the curve form itself being wrong (notably dilution:
new tracked players who have not played yet drag rates down).

## Validation (honest numbers)

The practice cases' histories are completely flat (0 of 55 trophies moved
across 7 reads), so no real-data backtest can discriminate models -- the
practice only checks that the program reads a case and writes a valid
forecast. On those cases this program's median equals the persistence
starter's exactly, and two runs are byte-identical.

So the model was validated on synthetic backtests instead: trophy-rate
evolution simulated under three different ground-truth shapes (cohort /
delayed-exponential, logit-linear in ln(1+t), piecewise-linear to an
asymptote), final-style cases (checkpoint Nov 21 11:30, target Nov 26,
~2.5 days of history). Mean pinball loss vs the persistence starter, 3 seeds
each:

| ground truth | this model | persistence | delta |
| --- | --- | --- | --- |
| cohort | 0.327 | 0.488 | -0.161 |
| logit-linear | 0.284 | 0.318 | -0.034 |
| piecewise | 0.358 | 0.318 | +0.040 |

It beats persistence under the two saturating / decelerating shapes and
loses narrowly under the linear-then-flat shape (the least realistic: real
trophy rates decelerate as the earner pool depletes and new players dilute
the tracked population). Coverage of the 90% interval is ~0.98 on the
synthetic finals (slightly wide, by design -- the final's 4.5-day
extrapolation is the uncertain part).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `<id>.in` in the
input directory and writes `<id>.out` to the output directory:
`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`.

---

Built with [make.fun](https://make.fun) — enter bounties yourself with
referral code `310ssgcx`: https://make.fun/r/310ssgcx
