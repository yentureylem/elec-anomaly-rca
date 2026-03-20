"""
data_generator.py
Synthetic electrical measurement generator with anomaly injection.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def generate_normal_signal(
    n_samples: int = 300,
    fs: float = 10.0,
    noise_level: float = 1.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a clean electrical measurement signal.

    Returns
    -------
    pd.DataFrame with columns: time, voltage, current, frequency, power_factor, thd
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n_samples) / fs

    voltage      = 230.0 + np.sin(t * 0.3) * 2.0  + np.random.normal(0, noise_level * 0.5, n_samples)
    current      = 10.0  + np.sin(t * 0.3 + 0.5) * 0.5 + np.random.normal(0, noise_level * 0.2, n_samples)
    frequency    = 50.0  + np.random.normal(0, 0.02, n_samples)
    power_factor = np.clip(0.95 + np.random.normal(0, 0.005, n_samples), 0.8, 1.0)
    thd          = 3.0   + np.random.normal(0, 0.3, n_samples)

    return pd.DataFrame({
        "time": t,
        "voltage": voltage,
        "current": current,
        "frequency": frequency,
        "power_factor": power_factor,
        "thd": thd,
    })


# ---------------------------------------------------------------------------
# Anomaly injectors
# ---------------------------------------------------------------------------

def inject_sensor_fault(
    data: pd.DataFrame,
    start_idx: int = 150,
    fault_type: str = "stuck",   # "stuck" | "offset" | "noise"
    affected_channel: str = "voltage",
) -> pd.DataFrame:
    """Sensor becomes stuck, offset, or noisy."""
    d = data.copy()
    n_tail = len(d) - start_idx

    if fault_type == "stuck":
        stuck = d[affected_channel].iloc[start_idx]
        d.loc[start_idx:, affected_channel] = stuck + np.random.normal(0, 0.05, n_tail)
    elif fault_type == "offset":
        offset = float(np.random.choice([-1, 1])) * np.random.uniform(5, 15)
        d.loc[start_idx:, affected_channel] += offset
    elif fault_type == "noise":
        d.loc[start_idx:, affected_channel] += np.random.normal(0, 8, n_tail)

    return d


def inject_cyber_attack(
    data: pd.DataFrame,
    start_idx: int = 100,
    end_idx: int = 160,
) -> pd.DataFrame:
    """Coordinated multi-channel data injection; breaks V–I correlation."""
    d = data.copy()
    dur = end_idx - start_idx

    d.loc[start_idx : end_idx - 1, "voltage"]      += np.random.uniform(-20, 20, dur)
    d.loc[start_idx : end_idx - 1, "current"]      += np.random.uniform(-5,  5,  dur)
    d.loc[start_idx : end_idx - 1, "frequency"]    += np.random.uniform(-0.5, 0.5, dur)
    d.loc[start_idx : end_idx - 1, "power_factor"]  = np.random.uniform(0.5, 1.0, dur)

    return d


def inject_equipment_fault(
    data: pd.DataFrame,
    start_idx: int = 100,
) -> pd.DataFrame:
    """Gradual degradation: voltage drop, PF decrease, current and THD rise."""
    d = data.copy()
    n_tail = len(d) - start_idx
    ramp = np.linspace(0, 1, n_tail)

    d.loc[start_idx:, "voltage"]      -= ramp * 15 + np.random.normal(0, 0.5, n_tail)
    d.loc[start_idx:, "power_factor"] -= ramp * 0.20
    d["power_factor"]                  = d["power_factor"].clip(0.5, 1.0)
    d.loc[start_idx:, "current"]      += ramp * 2  + np.random.normal(0, 0.1, n_tail)
    d.loc[start_idx:, "thd"]          += ramp * 8

    return d


# ---------------------------------------------------------------------------
# Scenario convenience wrapper
# ---------------------------------------------------------------------------

def get_scenario_data(scenario: str, seed: int = 42) -> Tuple[pd.DataFrame, int]:
    """Return (data, anomaly_start_idx) for a named scenario.

    scenario : "normal" | "sensor_fault" | "cyber_attack" | "equipment_fault"
    anomaly_start_idx : -1 for normal
    """
    np.random.seed(seed)
    base = generate_normal_signal(n_samples=300, seed=seed)

    if scenario == "normal":
        return base, -1

    elif scenario == "sensor_fault":
        ft    = np.random.choice(["stuck", "offset", "noise"])
        ch    = np.random.choice(["voltage", "current"])
        start = int(np.random.randint(120, 180))
        return inject_sensor_fault(base, start_idx=start, fault_type=ft, affected_channel=ch), start

    elif scenario == "cyber_attack":
        start = int(np.random.randint(80, 140))
        end   = int(np.clip(start + np.random.randint(30, 70), start + 20, 280))
        return inject_cyber_attack(base, start_idx=start, end_idx=end), start

    elif scenario == "equipment_fault":
        start = int(np.random.randint(80, 150))
        return inject_equipment_fault(base, start_idx=start), start

    return base, -1
