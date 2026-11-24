"""Vercel Serverless Function: Run a 1-hour simulation comparing Fixed vs AI."""

import json
import math
import os
import random
from http.server import BaseHTTPRequestHandler
from typing import Dict, List, Tuple

DATA_PATH = os.path.join(os.path.dirname(__file__), "model_data.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    MODEL_DATA = json.load(f)

Q_TABLE = MODEL_DATA["q_table"]
CFG = MODEL_DATA["config"]


def _poisson(lam: float, rng: random.Random) -> int:
    """Knuth Poisson sampler using Python stdlib only."""
    if lam <= 0.0:
        return 0
    if lam > 30.0:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def _bin_queue(q: int) -> int:
    if q <= 3:
        return 0
    if q <= 10:
        return 1
    if q <= 25:
        return 2
    return 3


def _bin_hour(hour: int) -> int:
    if 6 <= hour <= 9:
        return 1
    if 10 <= hour <= 15:
        return 2
    if 16 <= hour <= 21:
        return 3
    return 0


def run_one_hour(
    ew_hourly_raw: float, ns_hourly_raw: float, hour: int, mode: str, seed: int
) -> Dict:
    rng = random.Random(seed)
    step_s = CFG["step_seconds"]
    yellow_s = CFG["yellow_lost_seconds"]
    sat_rate = CFG["saturation_rate"]
    scale = CFG["demand_scale_factor"]
    min_green = CFG["min_green_seconds"]
    max_green = CFG["max_green_seconds"]
    fixed_green = CFG["fixed_green_seconds"]

    ew_rate = (ew_hourly_raw * scale) / 3600.0
    ns_rate = (ns_hourly_raw * scale) / 3600.0
    steps = 3600 // step_s

    q_ew, q_ns = 0, 0
    phase = 0  # 0 = EW green, 1 = NS green
    phase_elapsed = 0
    carry = 0.0

    total_wait_sec = 0.0
    total_q_sum = 0
    max_q = 0
    departed_total = 0
    timeline: List[Dict] = []

    h_bin = _bin_hour(hour)

    for s in range(steps):
        if mode == "fixed":
            action = 1 if phase_elapsed >= fixed_green else 0
        else:
            if phase_elapsed < min_green:
                action = 0
            elif phase_elapsed >= max_green:
                action = 1
            else:
                q_vals = Q_TABLE[_bin_queue(q_ew)][_bin_queue(q_ns)][phase][h_bin]
                action = 0 if q_vals[0] >= q_vals[1] else 1

        if action == 1:
            phase = 1 - phase
            phase_elapsed = step_s
            carry = 0.0
            eff_green = float(step_s - yellow_s)
        else:
            phase_elapsed += step_s
            eff_green = float(step_s)

        q_ew += _poisson(ew_rate * step_s, rng)
        q_ns += _poisson(ns_rate * step_s, rng)

        cap = eff_green * sat_rate + carry
        max_dep = int(cap)
        if phase == 0:
            dep = min(q_ew, max_dep)
            q_ew -= dep
            carry = (cap - dep) if q_ew > 0 else 0.0
        else:
            dep = min(q_ns, max_dep)
            q_ns -= dep
            carry = (cap - dep) if q_ns > 0 else 0.0

        q_tot = q_ew + q_ns
        total_q_sum += q_tot
        total_wait_sec += q_tot * step_s
        departed_total += dep
        if q_tot > max_q:
            max_q = q_tot

        # Sample every 12th step (once per minute) for smooth frontend playback
        if s % 6 == 0:
            timeline.append(
                {
                    "sec": (s + 1) * step_s,
                    "q_ew": q_ew,
                    "q_ns": q_ns,
                    "phase": phase,
                    "switched": action == 1,
                }
            )

    return {
        "avg_wait_sec": round(total_wait_sec / max(departed_total, 1), 2),
        "avg_queue": round(total_q_sum / steps, 2),
        "max_queue": max_q,
        "throughput": departed_total,
        "timeline": timeline,
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            params = json.loads(body.decode("utf-8"))
        except Exception:
            params = {}

        ew_vol = float(params.get("ew_volume", 4500))
        ns_vol = float(params.get("ns_volume", ew_vol * 0.40))
        hour = int(params.get("hour", 8)) % 24
        seed = int(params.get("seed", 42))

        fixed_res = run_one_hour(ew_vol, ns_vol, hour, "fixed", seed)
        ai_res = run_one_hour(ew_vol, ns_vol, hour, "ai", seed)

        resp = json.dumps({"fixed": fixed_res, "ai": ai_res}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
