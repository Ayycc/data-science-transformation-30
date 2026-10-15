"""SYNTHETIC hourly load and temperature for several balancing regions.

The generator writes in the structure the forecasting models are meant to find:

* daily and weekly load shapes (weekday vs weekend/holiday profiles),
* annual seasonality through temperature, with heating below 15 C and cooling above 20 C,
  applied to a thermally smoothed temperature (buildings lag the weather by a few hours),
* US federal holidays (the `holidays` package) and a Christmas-week dip,
* slow load growth, persistent (AR) weather anomalies shared partly across regions,
  AR(1) multiplicative noise,
* a regime shift: a data-centre cluster that adds 12% of base load in CENTRAL from April 2024,
* a day-ahead temperature *forecast* (actual + persistent error), which is what a forecaster would
  really have when scheduling a day ahead. Models only ever see the forecast for target hours.

Timestamps are hourly, naive local standard time (no daylight-saving shifts).
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from .config import REGIME_RAMP_DAYS, REGIME_REGION, REGIME_START, REGIME_UPLIFT, REGIONS, Region, Scale


def holiday_dates(start: str, end: str) -> set[pd.Timestamp]:
    """US federal holidays (observed) between start and end."""
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    us = holidays.country_holidays("US", years=list(years), observed=True)
    return {pd.Timestamp(d) for d in us.keys() if pd.Timestamp(start) <= pd.Timestamp(d) < pd.Timestamp(end)}


def _daily_anomaly(rng: np.random.Generator, n_days: int, phi: float, sd: float) -> np.ndarray:
    """Stationary AR(1) series of length n_days with marginal standard deviation sd."""
    eps = rng.normal(0.0, sd * np.sqrt(1 - phi**2), n_days)
    out = np.empty(n_days)
    out[0] = rng.normal(0.0, sd)
    for i in range(1, n_days):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def _hourly_from_daily(daily: np.ndarray, n_hours: int) -> np.ndarray:
    """Linear interpolation of a daily series (anchored at noon) onto hours."""
    x_day = np.arange(len(daily)) * 24 + 12
    return np.interp(np.arange(n_hours), x_day, daily)


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    eps = rng.normal(0.0, sd * np.sqrt(1 - phi**2), n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sd)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def _shape(hour: np.ndarray, off_day: np.ndarray, peak_shape: float) -> np.ndarray:
    """Relative daily load shape (mean about 1): night trough, morning ramp, daytime plateau, evening peak."""

    def bump(center: float, width: float) -> np.ndarray:
        d = np.minimum(np.abs(hour - center), 24 - np.abs(hour - center))
        return np.exp(-0.5 * (d / width) ** 2)

    weekday = 0.80 + 0.14 * bump(9.0, 2.0) + 0.18 * bump(14.0, 4.0) + 0.16 * peak_shape * bump(19.0, 2.2)
    weekend = 0.76 + 0.10 * bump(11.5, 3.0) + 0.10 * bump(15.0, 4.0) + 0.15 * peak_shape * bump(19.5, 2.4)
    return np.where(off_day, weekend, weekday)


def _smooth(x: np.ndarray, half_life_hours: float) -> np.ndarray:
    a = 1 - 0.5 ** (1 / half_life_hours)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + a * (x[i] - out[i - 1])
    return out


def generate(scale: Scale, temp_shift: tuple[str, str, float] | None = None) -> pd.DataFrame:
    """Return a long frame: region, ds, load_mw, temp_c, temp_fcst_c, is_holiday (all SYNTHETIC).

    temp_shift=(start, end, delta_c) adds delta_c to actual and forecast temperature in [start, end)
    for every region. The random draws are unchanged, so the result is the same synthetic world under a
    weather scenario (used for the temperature stress test).
    """
    rng = np.random.default_rng(scale.seed)
    ds = pd.date_range(scale.start, scale.end, freq="h", inclusive="left")
    n = len(ds)
    n_days = int(np.ceil(n / 24)) + 1
    hour = ds.hour.to_numpy().astype(float)
    doy = ds.dayofyear.to_numpy().astype(float)
    years = (ds - pd.Timestamp("2021-01-01")).total_seconds().to_numpy() / (365.25 * 86400)

    hol = holiday_dates(scale.start, scale.end)
    day = ds.normalize()
    is_holiday = day.isin(list(hol))
    off_day = (ds.dayofweek.to_numpy() >= 5) | is_holiday
    xmas = ((ds.month == 12) & (ds.day >= 24)) | ((ds.month == 1) & (ds.day <= 1))

    # Weather: shared continental anomaly plus regional anomalies.
    shared = _daily_anomaly(rng, n_days, phi=0.8, sd=2.6)
    regime_t0 = pd.Timestamp(REGIME_START)

    frames = []
    by_name = {r.name: r for r in REGIONS}
    for name in scale.regions:
        reg: Region = by_name[name]
        own = _daily_anomaly(rng, n_days, phi=0.75, sd=2.2)
        anomaly = _hourly_from_daily(0.6 * shared + 0.8 * own, n)
        temp = (
            reg.temp_mean
            + reg.temp_amp * np.cos(2 * np.pi * (doy - 200) / 365.25)
            + reg.diurnal_amp * np.cos(2 * np.pi * (hour - 15) / 24)
            + anomaly
            + _ar1(rng, n, phi=0.9, sd=0.6)
        )
        # Day-ahead temperature forecast error: persistent within a day, about 1.3 C.
        fc_err = _hourly_from_daily(_daily_anomaly(rng, n_days, phi=0.3, sd=1.1), n) + rng.normal(0, 0.5, n)
        if temp_shift is not None:
            in_window = (ds >= pd.Timestamp(temp_shift[0])) & (ds < pd.Timestamp(temp_shift[1]))
            temp = temp + np.where(in_window, temp_shift[2], 0.0)
        temp_fcst = temp + fc_err

        t_eff = _smooth(temp, half_life_hours=5.0)
        hdd = np.maximum(0.0, 15.0 - t_eff)
        cdd = np.maximum(0.0, t_eff - 20.0)
        growth = 1.0 + 0.012 * years
        shape = _shape(hour, off_day, reg.peak_shape)
        # Cooling bites hardest in the afternoon, heating in the morning and evening.
        cool_w = 0.7 + 0.5 * np.exp(-0.5 * ((hour - 16) / 4) ** 2)
        heat_w = 0.85 + 0.25 * np.exp(-0.5 * ((hour - 8) / 2.5) ** 2) + 0.2 * np.exp(-0.5 * ((hour - 19) / 2.5) ** 2)
        weather = reg.heat_mw_per_deg * heat_w * hdd + reg.cool_mw_per_deg * cool_w * cdd**1.25
        level = reg.base_mw * growth * (shape + weather)
        level = level * np.where(is_holiday, 0.95, 1.0) * np.where(xmas & ~is_holiday, 0.94, 1.0)
        if name == REGIME_REGION:
            ramp = np.clip((ds - regime_t0).total_seconds().to_numpy() / (REGIME_RAMP_DAYS * 86400), 0.0, 1.0)
            level = level + REGIME_UPLIFT * reg.base_mw * ramp
        noise = _ar1(rng, n, phi=0.9, sd=0.012)
        load = level * (1.0 + noise)
        frames.append(
            pd.DataFrame(
                {
                    "region": name,
                    "ds": ds,
                    "load_mw": np.round(load, 1),
                    "temp_c": np.round(temp, 2),
                    "temp_fcst_c": np.round(temp_fcst, 2),
                    "is_holiday": is_holiday.astype(np.int8),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
