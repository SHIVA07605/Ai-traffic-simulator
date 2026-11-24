"""
AI-Based Adaptive Traffic Signal Agent
======================================
Compares a tabular Q-learning adaptive traffic signal controller against a
fixed-timing (30s/30s) baseline on a simulated 4-way intersection driven by the
real-world UCI Metro Interstate Traffic Volume dataset (I-94, Minneapolis-St Paul).

Cross-Road Assumption:
    The UCI dataset provides hourly traffic volume for a single highway corridor
    (I-94). We treat `traffic_volume` as the main road (East-West) hourly demand
    and synthesize the cross road (North-South) hourly demand as 40% of the
    East-West volume plus small Gaussian noise (std = 5% of the 40% baseline,
    clipped at 0) using a fixed random seed for reproducibility.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Configuration & Constants
# =============================================================================
SEED = 42
CSV_FILENAME = "Metro_Interstate_Traffic_Volume.csv"
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/492/"
    "metro+interstate+traffic+volume.zip"
)
UCI_GZ_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/"
    "Metro_Interstate_Traffic_Volume.csv.gz"
)

# Intersection & Simulation Parameters
STEP_SECONDS = 5
STEPS_PER_HOUR = 3600 // STEP_SECONDS  # 720 steps/hour
YELLOW_LOST_SECONDS = 3
SATURATION_RATE = 0.5  # vehicles/second during green (~1 lane per direction)

# Scale multi-lane I-94 highway volume to a realistic 1-lane urban approach
# so peak hours (~6,500 veh/hr on I-94) approach intersection capacity.
DEMAND_SCALE_FACTOR = 0.18
NS_DEMAND_RATIO = 0.40  # Cross road = 40% of main road + small noise

# Signal Timing Constraints
FIXED_GREEN_SECONDS = 30
MIN_GREEN_SECONDS = 10
MAX_GREEN_SECONDS = 60

# Q-Learning Hyperparameters
NUM_EPISODES = 250
HOURS_PER_EPISODE = 24
ALPHA = 0.10
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.985
SWITCH_PENALTY = 2.0

# Peak Hours Definition: 7-9 AM and 4-6 PM
PEAK_HOURS = {7, 8, 9, 16, 17, 18}


# =============================================================================
# 1. Dataset Download & Preparation
# =============================================================================
def download_dataset(csv_path: str) -> None:
    """Download Metro Interstate Traffic Volume CSV from UCI if not present."""
    if os.path.exists(csv_path):
        print(f"[Data] Found existing dataset at '{csv_path}'.")
        return

    print("[Data] Downloading Metro Interstate Traffic Volume dataset from UCI...")
    # Try UCI static zip archive first
    try:
        req = urllib.request.Request(
            UCI_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            zip_bytes = resp.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                if name.endswith(".csv.gz"):
                    with zf.open(name) as gz_file, gzip.GzipFile(
                        fileobj=gz_file
                    ) as decompressed, open(csv_path, "wb") as out_f:
                        shutil.copyfileobj(decompressed, out_f)
                    print(f"[Data] Successfully extracted '{csv_path}'.")
                    return
                if name.endswith(".csv"):
                    with zf.open(name) as csv_file, open(
                        csv_path, "wb"
                    ) as out_f:
                        shutil.copyfileobj(csv_file, out_f)
                    print(f"[Data] Successfully extracted '{csv_path}'.")
                    return
    except Exception as exc:
        print(f"[Data] Primary UCI zip download failed ({exc}). Trying fallback...")

    # Fallback: direct .csv.gz URL
    try:
        req = urllib.request.Request(
            UCI_GZ_URL, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            gz_bytes = resp.read()
        with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes)) as decompressed, open(
            csv_path, "wb"
        ) as out_f:
            shutil.copyfileobj(decompressed, out_f)
        print(f"[Data] Successfully downloaded '{csv_path}'.")
        return
    except Exception as exc:
        print(f"[Data] Fallback download failed ({exc}).")

    raise RuntimeError(
        "Automatic download from UCI failed.\n"
        "Please download the dataset manually from:\n"
        "  https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume\n"
        f"and place the CSV file in this project folder as '{CSV_FILENAME}'."
    )


def load_and_prepare_data(
    csv_path: str, seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSV, clean timestamps/missing values, synthesize NS cross-road demand,
    convert to per-second arrival rates, and split 80/20 chronologically.
    """
    download_dataset(csv_path)

    df = pd.read_csv(csv_path)
    print(f"[Data] Loaded {len(df):,} raw rows.")

    # Parse date_time, sort chronologically, drop duplicate timestamps
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").drop_duplicates(
        subset=["date_time"], keep="first"
    )
    df = df.reset_index(drop=True)

    # Fill any missing values cleanly
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].ffill().bfill()

    # Main road (East-West) hourly volume
    ew_hourly = df["traffic_volume"].astype(float).to_numpy()

    # Cross road (North-South) assumption: 40% of EW volume + small random noise
    rng = np.random.default_rng(seed)
    noise_std = 0.05 * (NS_DEMAND_RATIO * ew_hourly)
    ns_hourly = np.clip(
        NS_DEMAND_RATIO * ew_hourly + rng.normal(0.0, noise_std),
        a_min=0.0,
        a_max=None,
    )

    # Scale to a 1-lane-per-direction intersection and convert to veh/sec
    df["ew_rate"] = (ew_hourly * DEMAND_SCALE_FACTOR) / 3600.0
    df["ns_rate"] = (ns_hourly * DEMAND_SCALE_FACTOR) / 3600.0
    df["hour"] = df["date_time"].dt.hour
    df["is_peak"] = df["hour"].isin(PEAK_HOURS)

    # Chronological 80% / 20% split (no shuffling)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print(
        f"[Data] Cleaned unique hours: {len(df):,} | "
        f"Train: {len(train_df):,} hrs | Test: {len(test_df):,} hrs"
    )
    return train_df, test_df


# =============================================================================
# 2. Intersection Simulation Environment
# =============================================================================
class IntersectionSim:
    """
    4-way intersection with 2 phases:
        Phase 0 (A): East-West green
        Phase 1 (B): North-South green
    Each step represents STEP_SECONDS (5 s). Phase switches incur
    YELLOW_LOST_SECONDS (3 s) of lost time during the transition step.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.q_ew: int = 0
        self.q_ns: int = 0
        self.phase: int = 0  # 0 = EW green, 1 = NS green
        self.phase_elapsed: int = 0
        self._frac_carry: float = 0.0

    def step(
        self, action: int, ew_rate: float, ns_rate: float, rng: np.random.Generator
    ) -> Tuple[int, float]:
        """
        Advance simulation by 5 seconds.
        action: 0 = keep current phase, 1 = switch phase.
        Returns (departed_vehicles, step_reward).
        """
        switched = action == 1
        if switched:
            self.phase = 1 - self.phase
            self.phase_elapsed = STEP_SECONDS
            self._frac_carry = 0.0
            effective_green = float(STEP_SECONDS - YELLOW_LOST_SECONDS)
        else:
            self.phase_elapsed += STEP_SECONDS
            effective_green = float(STEP_SECONDS)

        # Poisson arrivals for this 5-second step
        arr_ew = int(rng.poisson(ew_rate * STEP_SECONDS))
        arr_ns = int(rng.poisson(ns_rate * STEP_SECONDS))
        self.q_ew += arr_ew
        self.q_ns += arr_ns

        # Departures at saturation flow during effective green
        capacity = effective_green * SATURATION_RATE + self._frac_carry
        max_depart = int(capacity)

        if self.phase == 0:
            departed = min(self.q_ew, max_depart)
            self.q_ew -= departed
            self._frac_carry = (capacity - departed) if self.q_ew > 0 else 0.0
        else:
            departed = min(self.q_ns, max_depart)
            self.q_ns -= departed
            self._frac_carry = (capacity - departed) if self.q_ns > 0 else 0.0

        total_queue = self.q_ew + self.q_ns
        reward = -float(total_queue) - (SWITCH_PENALTY if switched else 0.0)
        return departed, reward


# =============================================================================
# 3. Controllers (Fixed-Timing Baseline & Tabular Q-Learning Agent)
# =============================================================================
class FixedTimingController:
    """Fixed 30s East-West green, 30s North-South green, repeating."""

    def select_action(self, sim: IntersectionSim) -> int:
        if sim.phase_elapsed >= FIXED_GREEN_SECONDS:
            return 1
        return 0


def _bin_queue(q: int) -> int:
    """Discretize queue length into 4 bins: low, medium, high, very high."""
    if q <= 3:
        return 0
    if q <= 10:
        return 1
    if q <= 25:
        return 2
    return 3


def _bin_hour(hour: int) -> int:
    """Discretize hour-of-day into 4 buckets: night, morning peak, midday, evening peak."""
    if 6 <= hour <= 9:
        return 1  # Morning peak
    if 10 <= hour <= 15:
        return 2  # Midday
    if 16 <= hour <= 21:
        return 3  # Evening peak
    return 0  # Night (22-5)


class QLearningAgent:
    """Tabular Q-learning traffic signal agent."""

    def __init__(self, seed: int = SEED) -> None:
        # State: (ew_bin[4], ns_bin[4], phase[2], hour_bucket[4]) -> Action[2]
        self.q_table = np.zeros((4, 4, 2, 4, 2), dtype=np.float64)
        self.epsilon = EPSILON_START
        self.rng = np.random.default_rng(seed)

    def get_state(self, sim: IntersectionSim, hour: int) -> Tuple[int, int, int, int]:
        return (
            _bin_queue(sim.q_ew),
            _bin_queue(sim.q_ns),
            sim.phase,
            _bin_hour(hour),
        )

    def select_action(
        self, sim: IntersectionSim, hour: int, greedy: bool = False
    ) -> Tuple[int, bool]:
        """
        Choose action (0=keep, 1=switch) respecting min/max green constraints.
        Returns (action, can_choose) where can_choose is False if forced.
        """
        if sim.phase_elapsed < MIN_GREEN_SECONDS:
            return 0, False
        if sim.phase_elapsed >= MAX_GREEN_SECONDS:
            return 1, False

        state = self.get_state(sim, hour)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, 2)), True

        q_vals = self.q_table[state]
        if q_vals[0] == q_vals[1]:
            return 0, True  # Prefer keeping phase on tie
        return int(np.argmax(q_vals)), True

    def update(
        self,
        state: Tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int],
    ) -> None:
        best_next = float(np.max(self.q_table[next_state]))
        td_target = reward + GAMMA * best_next
        self.q_table[state][action] += ALPHA * (
            td_target - self.q_table[state][action]
        )

    def decay_epsilon(self) -> None:
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


# =============================================================================
# 4. Training & Evaluation Routines
# =============================================================================
def train_agent(train_df: pd.DataFrame, seed: int = SEED) -> Tuple[QLearningAgent, List[float]]:
    """Train tabular Q-learning agent on 24-hour episodes from the training split."""
    agent = QLearningAgent(seed=seed)
    sim_rng = np.random.default_rng(seed + 1)
    sim = IntersectionSim()

    ew_rates = train_df["ew_rate"].to_numpy()
    ns_rates = train_df["ns_rate"].to_numpy()
    hours = train_df["hour"].to_numpy()
    max_start = len(train_df) - HOURS_PER_EPISODE

    episode_rewards: List[float] = []
    print(f"[Train] Training Q-learning agent for {NUM_EPISODES} episodes...")

    for ep in range(NUM_EPISODES):
        start_idx = int(sim_rng.integers(0, max_start))
        sim.reset()
        ep_reward = 0.0

        for h_offset in range(HOURS_PER_EPISODE):
            idx = start_idx + h_offset
            ew_r = float(ew_rates[idx])
            ns_r = float(ns_rates[idx])
            hr = int(hours[idx])

            for _ in range(STEPS_PER_HOUR):
                state = agent.get_state(sim, hr)
                action, can_choose = agent.select_action(sim, hr, greedy=False)
                _, reward = sim.step(action, ew_r, ns_r, sim_rng)
                next_state = agent.get_state(sim, hr)

                if can_choose:
                    agent.update(state, action, reward, next_state)
                ep_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(ep_reward)

        if (ep + 1) % 50 == 0:
            avg_recent = np.mean(episode_rewards[-25:])
            print(
                f"  Episode {ep + 1:3d}/{NUM_EPISODES} | "
                f"Avg Recent Reward: {avg_recent:10.1f} | Epsilon: {agent.epsilon:.3f}"
            )

    return agent, episode_rewards


@dataclass
class EvalMetrics:
    avg_wait_sec: float
    avg_queue: float
    max_queue: int
    throughput_per_hr: float
    peak_avg_wait_sec: float
    peak_avg_queue: float
    offpeak_avg_wait_sec: float
    offpeak_avg_queue: float
    hourly_avg_queues: np.ndarray


def evaluate_controller(
    test_df: pd.DataFrame,
    controller: FixedTimingController | QLearningAgent,
    seed: int = SEED,
) -> EvalMetrics:
    """Evaluate a controller across the entire test split using a fixed seed."""
    rng = np.random.default_rng(seed)
    sim = IntersectionSim()
    is_agent = isinstance(controller, QLearningAgent)

    ew_rates = test_df["ew_rate"].to_numpy()
    ns_rates = test_df["ns_rate"].to_numpy()
    hours = test_df["hour"].to_numpy()
    is_peak_arr = test_df["is_peak"].to_numpy()

    total_wait_veh_sec = 0.0
    total_queue_sum = 0
    total_steps = 0
    total_departed = 0
    max_queue = 0

    peak_wait_veh_sec = 0.0
    peak_queue_sum = 0
    peak_steps = 0
    peak_departed = 0

    off_wait_veh_sec = 0.0
    off_queue_sum = 0
    off_steps = 0
    off_departed = 0

    hourly_avg_queues = np.zeros(len(test_df), dtype=np.float64)

    for i in range(len(test_df)):
        ew_r = float(ew_rates[i])
        ns_r = float(ns_rates[i])
        hr = int(hours[i])
        peak = bool(is_peak_arr[i])

        hr_queue_sum = 0
        hr_departed = 0

        for _ in range(STEPS_PER_HOUR):
            if is_agent:
                action, _ = controller.select_action(sim, hr, greedy=True)
            else:
                action = controller.select_action(sim)

            departed, _ = sim.step(action, ew_r, ns_r, rng)
            q_tot = sim.q_ew + sim.q_ns

            hr_queue_sum += q_tot
            hr_departed += departed
            if q_tot > max_queue:
                max_queue = q_tot

        hourly_avg_queues[i] = hr_queue_sum / STEPS_PER_HOUR

        wait_sec = hr_queue_sum * STEP_SECONDS
        total_wait_veh_sec += wait_sec
        total_queue_sum += hr_queue_sum
        total_steps += STEPS_PER_HOUR
        total_departed += hr_departed

        if peak:
            peak_wait_veh_sec += wait_sec
            peak_queue_sum += hr_queue_sum
            peak_steps += STEPS_PER_HOUR
            peak_departed += hr_departed
        else:
            off_wait_veh_sec += wait_sec
            off_queue_sum += hr_queue_sum
            off_steps += STEPS_PER_HOUR
            off_departed += hr_departed

    return EvalMetrics(
        avg_wait_sec=total_wait_veh_sec / max(total_departed, 1),
        avg_queue=total_queue_sum / total_steps,
        max_queue=max_queue,
        throughput_per_hr=total_departed / len(test_df),
        peak_avg_wait_sec=peak_wait_veh_sec / max(peak_departed, 1),
        peak_avg_queue=peak_queue_sum / max(peak_steps, 1),
        offpeak_avg_wait_sec=off_wait_veh_sec / max(off_departed, 1),
        offpeak_avg_queue=off_queue_sum / max(off_steps, 1),
        hourly_avg_queues=hourly_avg_queues,
    )


# =============================================================================
# 5. Reporting & Plotting
# =============================================================================
def _pct_improvement(baseline: float, agent: float, higher_is_better: bool = False) -> float:
    if baseline == 0:
        return 0.0
    if higher_is_better:
        return ((agent - baseline) / baseline) * 100.0
    return ((baseline - agent) / baseline) * 100.0


def print_comparison(fixed: EvalMetrics, ai: EvalMetrics) -> None:
    print("\n" + "=" * 72)
    print("EVALUATION RESULTS (Test Split: Last 20% Chronological)")
    print("=" * 72)
    rows = [
        ("Avg Waiting Time (s/veh)", fixed.avg_wait_sec, ai.avg_wait_sec, False),
        ("Avg Queue Length (veh)", fixed.avg_queue, ai.avg_queue, False),
        ("Max Queue Length (veh)", float(fixed.max_queue), float(ai.max_queue), False),
        ("Throughput (veh/hour)", fixed.throughput_per_hr, ai.throughput_per_hr, True),
        ("Peak Avg Wait (s/veh)", fixed.peak_avg_wait_sec, ai.peak_avg_wait_sec, False),
        ("Peak Avg Queue (veh)", fixed.peak_avg_queue, ai.peak_avg_queue, False),
        ("Off-Peak Avg Wait (s/veh)", fixed.offpeak_avg_wait_sec, ai.offpeak_avg_wait_sec, False),
        ("Off-Peak Avg Queue (veh)", fixed.offpeak_avg_queue, ai.offpeak_avg_queue, False),
    ]
    print(f"{'Metric':<28} {'Fixed-Timing':>14} {'AI Agent':>14} {'Improvement':>13}")
    print("-" * 72)
    for label, f_val, a_val, hib in rows:
        imp = _pct_improvement(f_val, a_val, higher_is_better=hib)
        print(f"{label:<28} {f_val:>14.2f} {a_val:>14.2f} {imp:>+12.2f}%")
    print("=" * 72 + "\n")


def save_plots(
    episode_rewards: List[float],
    fixed: EvalMetrics,
    ai: EvalMetrics,
    test_df: pd.DataFrame,
    out_dir: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Training Reward Curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    episodes = np.arange(1, len(episode_rewards) + 1)
    ax.plot(episodes, episode_rewards, alpha=0.35, color="#4C72B0", label="Episode Reward")
    window = 15
    if len(episode_rewards) >= window:
        ma = pd.Series(episode_rewards).rolling(window=window).mean()
        ax.plot(episodes, ma, color="#C44E52", linewidth=2.0, label=f"{window}-Ep Moving Avg")
    ax.set_title("Q-Learning Training Reward Curve")
    ax.set_xlabel("Episode (24 Hours Each)")
    ax.set_ylabel("Total Episode Reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_reward_curve.png"), dpi=150)
    plt.close(fig)

    # 2. Average Waiting Time: Fixed vs AI (Overall, Peak, Off-Peak)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    categories = ["Overall", "Peak (7-9 AM, 4-6 PM)", "Off-Peak"]
    fixed_vals = [fixed.avg_wait_sec, fixed.peak_avg_wait_sec, fixed.offpeak_avg_wait_sec]
    ai_vals = [ai.avg_wait_sec, ai.peak_avg_wait_sec, ai.offpeak_avg_wait_sec]
    x = np.arange(len(categories))
    width = 0.35
    b1 = ax.bar(x - width / 2, fixed_vals, width, label="Fixed-Timing (30s/30s)", color="#DD8452")
    b2 = ax.bar(x + width / 2, ai_vals, width, label="AI Q-Learning Agent", color="#55A868")
    ax.bar_label(b1, fmt="%.1f s", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.1f s", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Average Waiting Time (seconds / vehicle)")
    ax.set_title("Average Vehicle Waiting Time: Fixed-Timing vs AI Agent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "avg_waiting_time_comparison.png"), dpi=150)
    plt.close(fig)

    # 3. Queue Length over a 24-Hour Sample Day (first complete 00:00-23:00 day in test)
    zero_indices = test_df.index[test_df["hour"] == 0].tolist()
    sample_start = zero_indices[0] if zero_indices and zero_indices[0] + 24 <= len(test_df) else 0
    sample_slice = slice(sample_start, sample_start + 24)
    sample_hours = np.arange(24)
    sample_date = test_df["date_time"].iloc[sample_start].strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(
        sample_hours,
        fixed.hourly_avg_queues[sample_slice],
        marker="o",
        linewidth=2.0,
        color="#DD8452",
        label="Fixed-Timing (30s/30s)",
    )
    ax.plot(
        sample_hours,
        ai.hourly_avg_queues[sample_slice],
        marker="s",
        linewidth=2.0,
        color="#55A868",
        label="AI Q-Learning Agent",
    )
    ax.set_title(f"Average Queue Length Over 24-Hour Sample Day ({sample_date})")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Total Queue Length (vehicles)")
    ax.set_xticks(sample_hours)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "queue_length_24h_sample.png"), dpi=150)
    plt.close(fig)

    print(f"[Output] Saved plots to '{out_dir}':")
    print("  - training_reward_curve.png")
    print("  - avg_waiting_time_comparison.png")
    print("  - queue_length_24h_sample.png")


# =============================================================================
# Main Entrypoint
# =============================================================================
def main() -> None:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(project_dir, CSV_FILENAME)

    try:
        train_df, test_df = load_and_prepare_data(csv_path, seed=SEED)
    except RuntimeError as err:
        print(f"\nERROR: {err}")
        return

    agent, episode_rewards = train_agent(train_df, seed=SEED)

    fixed_controller = FixedTimingController()
    print("[Eval] Evaluating Fixed-Timing baseline on test split...")
    fixed_metrics = evaluate_controller(test_df, fixed_controller, seed=SEED + 100)

    print("[Eval] Evaluating AI Q-Learning Agent on test split...")
    ai_metrics = evaluate_controller(test_df, agent, seed=SEED + 100)

    print_comparison(fixed_metrics, ai_metrics)
    save_plots(episode_rewards, fixed_metrics, ai_metrics, test_df, project_dir)
    export_web_artifacts(
        agent, episode_rewards, fixed_metrics, ai_metrics, test_df, project_dir
    )


def export_web_artifacts(
    agent: QLearningAgent,
    episode_rewards: List[float],
    fixed: EvalMetrics,
    ai: EvalMetrics,
    test_df: pd.DataFrame,
    out_dir: str,
) -> None:
    """Export trained Q-table, metrics, and 24h sample data for the web UI & Vercel API."""
    zero_indices = test_df.index[test_df["hour"] == 0].tolist()
    sample_start = (
        zero_indices[0]
        if zero_indices and zero_indices[0] + 24 <= len(test_df)
        else 0
    )
    sample_slice = slice(sample_start, sample_start + 24)
    sample_df = test_df.iloc[sample_slice]

    payload = {
        "q_table": agent.q_table.tolist(),
        "config": {
            "step_seconds": STEP_SECONDS,
            "yellow_lost_seconds": YELLOW_LOST_SECONDS,
            "saturation_rate": SATURATION_RATE,
            "min_green_seconds": MIN_GREEN_SECONDS,
            "max_green_seconds": MAX_GREEN_SECONDS,
            "fixed_green_seconds": FIXED_GREEN_SECONDS,
            "demand_scale_factor": DEMAND_SCALE_FACTOR,
            "ns_demand_ratio": NS_DEMAND_RATIO,
        },
        "training_rewards": [round(float(r), 1) for r in episode_rewards],
        "evaluation": {
            "fixed": {
                "avg_wait_sec": round(fixed.avg_wait_sec, 2),
                "avg_queue": round(fixed.avg_queue, 2),
                "max_queue": int(fixed.max_queue),
                "throughput_per_hr": round(fixed.throughput_per_hr, 2),
                "peak_avg_wait_sec": round(fixed.peak_avg_wait_sec, 2),
                "peak_avg_queue": round(fixed.peak_avg_queue, 2),
                "offpeak_avg_wait_sec": round(fixed.offpeak_avg_wait_sec, 2),
                "offpeak_avg_queue": round(fixed.offpeak_avg_queue, 2),
            },
            "ai": {
                "avg_wait_sec": round(ai.avg_wait_sec, 2),
                "avg_queue": round(ai.avg_queue, 2),
                "max_queue": int(ai.max_queue),
                "throughput_per_hr": round(ai.throughput_per_hr, 2),
                "peak_avg_wait_sec": round(ai.peak_avg_wait_sec, 2),
                "peak_avg_queue": round(ai.peak_avg_queue, 2),
                "offpeak_avg_wait_sec": round(ai.offpeak_avg_wait_sec, 2),
                "offpeak_avg_queue": round(ai.offpeak_avg_queue, 2),
            },
        },
        "sample_day": {
            "date": test_df["date_time"].iloc[sample_start].strftime("%Y-%m-%d"),
            "hours": [int(h) for h in sample_df["hour"].tolist()],
            "ew_volumes": [round(float(v), 1) for v in sample_df["traffic_volume"].tolist()],
            "ew_rates": [round(float(r), 5) for r in sample_df["ew_rate"].tolist()],
            "ns_rates": [round(float(r), 5) for r in sample_df["ns_rate"].tolist()],
            "fixed_queues": [round(float(q), 2) for q in fixed.hourly_avg_queues[sample_slice]],
            "ai_queues": [round(float(q), 2) for q in ai.hourly_avg_queues[sample_slice]],
        },
    }

    os.makedirs(os.path.join(out_dir, "public"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "api"), exist_ok=True)
    for dest in (
        os.path.join(out_dir, "public", "model_data.json"),
        os.path.join(out_dir, "api", "model_data.json"),
    ):
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    print("[Output] Exported 'public/model_data.json' and 'api/model_data.json' for UI & Vercel.")


if __name__ == "__main__":
    main()
