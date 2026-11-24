# AI-Based Adaptive Traffic Signal Agent

An adaptive traffic signal controller using **tabular Q-learning** compared against a **fixed-timing (30s/30s)** baseline on a simulated 4-way intersection driven by real-world hourly traffic data from the **UCI Metro Interstate Traffic Volume** dataset (I-94, Minneapolis–St Paul).

## Dataset & Assumptions
- **Source**: [UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) (~48,000 hourly records).
- `main.py` automatically downloads and extracts `Metro_Interstate_Traffic_Volume.csv` from UCI. If the download fails, place `Metro_Interstate_Traffic_Volume.csv` in this folder and re-run.
- **Cross-Road Assumption**: Because the UCI dataset only records volume along I-94, `traffic_volume` is used as the main road (**East-West**) hourly demand, and the cross road (**North-South**) hourly demand is synthesized as **40% of East-West volume plus small Gaussian noise** (std = 5% of the 40% baseline, clipped at 0) with a fixed random seed (`42`).
- **Intersection Scaling**: Multi-lane I-94 highway volumes are scaled (`0.18`) and converted to vehicles/second (`volume / 3600`) to represent a realistic 1-lane-per-direction urban intersection with a saturation flow rate of `0.5` vehicles/second during green.
- **Train/Test Split**: Chronological 80% training / 20% testing (no shuffling).

## Simulation & Agent Design
- **Simulation**: 5-second discrete time steps, Poisson vehicle arrivals, 2 phases (EW green / NS green), and 3-second yellow/lost time on phase switches.
- **Baseline**: Fixed 30 s EW green, 30 s NS green, repeating.
- **Q-Learning Agent**:
  - **State**: `(EW queue bin, NS queue bin, current phase, hour-of-day bucket)` — 128 states total.
  - **Actions**: Keep current phase or switch phase, subject to a **10 s minimum green** and **60 s maximum green**.
  - **Reward**: Negative total queue length at each step minus a small switching penalty.

## Setup & Usage

```bash
pip install -r requirements.txt
python main.py
```

## Outputs
Running `main.py` prints a full comparison table (overall, peak hours 7–9 AM & 4–6 PM, and off-peak hours) and saves three plots:
- `training_reward_curve.png`
- `avg_waiting_time_comparison.png`
- `queue_length_24h_sample.png`
