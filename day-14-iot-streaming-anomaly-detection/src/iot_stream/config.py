"""Configuration dataclasses. Every random process is seeded from these values."""

from __future__ import annotations

from dataclasses import dataclass, field

SENSORS: tuple[str, ...] = ("vibration", "temperature", "current")
HOUR = 3600.0
DAY = 86400.0


@dataclass(frozen=True)
class SimConfig:
    """SYNTHETIC fleet + sensor stream generator settings."""

    n_machines: int = 500
    days: float = 14.0
    tick_s: float = 300.0  # one reading per sensor per machine every 5 minutes
    seed: int = 20261024
    # lifecycle / degradation
    weibull_shape: float = 2.0
    weibull_scale_days: float = 16.0
    degradation_hours: tuple[float, float] = (24.0, 96.0)
    sudden_failure_share: float = 0.12
    downtime_hours: tuple[float, float] = (2.0, 8.0)
    # drift
    seasonal_warming_c: float = 5.0  # gradual ambient warming over the full horizon
    regime_change_day: float = 8.0  # abrupt production change (new product)
    regime_vib_mult: float = 1.30
    regime_cur_mult: float = 1.18
    regime_temp_add: float = 3.0
    calib_drift_share: float = 0.05  # machines whose temperature sensor drifts
    calib_drift_c_per_day: float = 0.4
    # data quality
    dropouts_per_sensor_per_day: float = 0.25
    dropout_minutes: tuple[float, float] = (10.0, 240.0)
    point_loss: float = 0.003
    # transport delays (arrival = event time + delay)
    late_share: float = 0.07  # moderately late (lognormal, median 2 min)
    gateway_outages_per_machine_per_day: float = 0.15  # store-and-forward bursts
    gateway_outage_minutes: tuple[float, float] = (15.0, 120.0)


@dataclass(frozen=True)
class StreamConfig:
    """Event-time windowing settings."""

    slide_s: float = 1800.0  # 30 min panes / tumbling windows / slide of the sliding window
    sliding_size_s: float = 7200.0  # 2 h sliding window
    allowed_lateness_s: float = 600.0  # watermark = max event time - 10 min
    batch_size: int = 4096  # events per micro-batch (Kafka-poll analogue)


@dataclass(frozen=True)
class ModelConfig:
    """Online models, drift detectors and warning rules."""

    seed: int = 7
    baseline_windows: int = 6  # 3 h of healthy data after each (re)start
    hst_trees: int = 15
    hst_height: int = 8
    hst_window: int = 1000
    hst_threshold: float = 0.95
    hst_persist: int = 2
    rul_leaf: str = "adaptive"  # HoeffdingTreeRegressor leaf_prediction (mean / model / adaptive)
    rul_cap_h: float = 72.0
    rul_warn_h: float = 24.0
    rul_persist: int = 2
    rul_min_labels: int = 3000
    rul_min_cap_labels: int = 10000
    adwin_delta: float = 0.002
    ph_delta: float = 10.0
    ph_threshold: float = 50000.0
    ph_min_instances: int = 2000
    debounce_h: float = 24.0
    eval_horizon_h: float = 72.0  # warning is a true positive if the machine fails within this horizon
    warmup_h: float = 24.0  # failures before this time are excluded from recall (system warm-up)
    static_freeze_day: float = 5.0  # "batch-like" baseline variant stops learning here
    hst_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "vib_rel": (-0.4, 1.6),
            "vib_2h_rel": (-0.4, 1.6),
            "vib_std_rel": (-1.0, 4.0),
            "vib_slope": (-0.3, 0.3),
            "cur_rel": (-0.25, 0.45),
            "temp_d": (-0.6, 1.6),
            "temp_slope": (-0.3, 0.3),
        }
    )
