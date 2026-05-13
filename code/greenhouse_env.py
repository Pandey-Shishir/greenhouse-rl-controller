"""
greenhouse_env.py
=================
Phase 3: Custom Gymnasium environment for greenhouse RL controller.

What this file does:
    Simulates a greenhouse that an AI agent learns to control.
    The agent sees 7 sensor readings (state), picks one of 6 actions,
    and receives a reward based on how close conditions are to optimal.

    All parameter ranges come from real data in parameter_ranges.json:
        - greenhouse_sensor_10min.csv  (Marcel Boonman, 17,562 rows)
        - plant_growth_conditions.csv  (193 rows, growth milestone labels)

How to run:
    pip install gymnasium numpy
    python greenhouse_env.py

How to use in Colab / training:
    from greenhouse_env import GreenhouseEnv
    env = GreenhouseEnv()
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ─── Parameter ranges from parameter_ranges.json ─────────────────────────────
# Each variable: [sim_min, sim_max, optimal_min, optimal_max]
# Source: greenhouse_sensor_10min.csv + plant_growth_conditions.csv

PARAMS = {
    "temperature_c":    [5.0,   45.0,  18.0,  26.0],
    "humidity_pct":     [20.0,  85.0,  50.0,  70.0],
    "light_lux":        [0.0,   35000, 3000,  10000],
    "co2_ppm":          [400.0, 2000.0, 800.0, 1500.0],
    "soil_moisture_pct":[10.0,  90.0,  40.0,  70.0],
    "voc_ppb":          [0.0,   2000.0, 0.0,  500.0],
    # time_of_day: 0 to 23 hours (natural cycle, not controlled by agent)
    "time_of_day":      [0.0,   23.0,  6.0,   20.0],
}

PARAM_NAMES = list(PARAMS.keys())
N_VARS = len(PARAM_NAMES)  # 7

# ─── Action space ─────────────────────────────────────────────────────────────
# 6 discrete actions. Each action maps to what an ESP32 relay board would do.
# Index: name, description
ACTIONS = {
    0: "do_nothing",     # No actuator fired. Natural drift only.
    1: "open_vents",     # Ventilation fan on. Cools and dries the air.
    2: "water_plants",   # Irrigation pump on. Raises soil moisture and humidity.
    3: "grow_lights",    # Grow lights on. Raises lux and slightly raises temp.
    4: "heater",         # Heater on. Raises temperature, slightly lowers humidity.
    5: "inject_co2",     # CO2 valve open. Raises CO2 concentration.
}
N_ACTIONS = len(ACTIONS)  # 6

# ─── Energy cost per action ───────────────────────────────────────────────────
# Higher cost = more electricity used. Agent is penalised for energy waste.
ENERGY_COST = {
    0: 0.0,   # Free
    1: 0.1,   # Fan: low cost
    2: 0.2,   # Pump: moderate
    3: 0.5,   # Grow lights: significant
    4: 0.8,   # Heater: expensive
    5: 0.3,   # CO2 injection: moderate
}

# ─── Actuator physics ─────────────────────────────────────────────────────────
# How each action changes each variable per step (one step = 10 minutes).
# These are calibrated to match the greenhouse_sensor_10min.csv dynamics.
# Format: {action_index: {variable_name: delta_per_step}}

ACTUATOR_EFFECTS = {
    0: {},  # do nothing: no direct effect
    1: {    # open_vents
        "temperature_c":    -1.5,   # cooling
        "humidity_pct":     -3.0,   # drying
        "co2_ppm":          -60.0,  # CO2 escapes
        "voc_ppb":          -150.0, # VOC escapes
    },
    2: {    # water_plants
        "soil_moisture_pct": 15.0,  # soil absorbs water
        "humidity_pct":       5.0,  # evaporation raises humidity
    },
    3: {    # grow_lights
        "light_lux":        5000.0, # lights add lux
        "temperature_c":       0.5, # lights generate heat
    },
    4: {    # heater
        "temperature_c":     2.0,   # direct heat
        "humidity_pct":     -1.0,   # warm air holds less relative humidity
    },
    5: {    # inject_co2
        "co2_ppm":          150.0,  # direct CO2 increase
    },
}


class GreenhouseEnv(gym.Env):
    """
    Custom Gymnasium environment simulating a greenhouse.

    Observation space: 7 continuous variables, each normalised to [0, 1].
        [temperature, humidity, light, co2, soil_moisture, voc, time_of_day]

    Action space: 6 discrete actions (see ACTIONS above).

    Reward: +1 per variable in optimal range, -penalty for out-of-range,
            -energy cost per action, +2 bonus if ALL variables optimal.

    Episode length: 144 steps = 24 hours at 10-minute intervals.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Observation: 7 normalised floats in [0, 1]
        self.observation_space = spaces.Box(
            low=np.zeros(N_VARS, dtype=np.float32),
            high=np.ones(N_VARS, dtype=np.float32),
            dtype=np.float32,
        )

        # Action: one integer 0-5
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Episode length: 144 steps (24 hours at 10-min intervals)
        self.max_steps = 144
        self._step_count = 0
        self._state = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _raw_to_norm(self, raw_state):
        """Convert raw state values to normalised [0,1] for the agent."""
        norm = np.zeros(N_VARS, dtype=np.float32)
        for i, name in enumerate(PARAM_NAMES):
            lo, hi, _, _ = PARAMS[name]
            norm[i] = np.clip((raw_state[i] - lo) / (hi - lo), 0.0, 1.0)
        return norm

    def _compute_reward(self, raw_state, action):
        """
        Reward function grounded in real data optimal ranges.

        Per-variable score:
            +1.0  if value is inside [optimal_min, optimal_max]
            Scaled penalty (down to -1.0) proportional to how far outside
        Energy penalty for using actuators.
        Bonus +2.0 if every variable is simultaneously in optimal range.
        """
        total = 0.0
        all_optimal = True

        for i, name in enumerate(PARAM_NAMES):
            lo, hi, opt_lo, opt_hi = PARAMS[name]
            val = raw_state[i]

            if opt_lo <= val <= opt_hi:
                total += 1.0
            else:
                # Distance from nearest optimal boundary, normalised
                if val < opt_lo:
                    dist = (opt_lo - val) / max(opt_lo - lo, 1e-6)
                else:
                    dist = (val - opt_hi) / max(hi - opt_hi, 1e-6)
                penalty = -np.clip(dist, 0.0, 1.0)
                total += penalty
                all_optimal = False

        if all_optimal:
            total += 2.0

        # Energy cost
        total -= ENERGY_COST[action]

        return float(total)

    def _natural_drift(self, raw_state):
        """
        Variables drift naturally each step without any action.
        Represents passive physics: evaporation, plant CO2 uptake,
        heat exchange with outside air, and sensor noise.
        """
        state = raw_state.copy()
        rng = self.np_random

        # Temperature drifts toward 20°C baseline (indoor equilibrium)
        state[0] += (20.0 - state[0]) * 0.03 + rng.normal(0, 0.3)

        # Humidity drifts toward 55% baseline
        state[1] += (55.0 - state[1]) * 0.02 + rng.normal(0, 0.5)

        # Light follows a smooth day/night sine curve
        hour = state[6]
        if 6 <= hour <= 20:
            target_lux = 8000 * np.sin(np.pi * (hour - 6) / 14)
        else:
            target_lux = 0.0
        state[2] += (target_lux - state[2]) * 0.15 + rng.normal(0, 50)

        # CO2 decreases as plants photosynthesise during daylight
        if 6 <= hour <= 20:
            state[3] -= 8.0 + rng.normal(0, 2)
        else:
            state[3] += 2.0  # slight rise at night (respiration)

        # Soil moisture decreases from evapotranspiration
        state[4] -= 0.8 + rng.normal(0, 0.2)

        # VOC builds slowly, decays slowly
        state[5] += rng.normal(0, 20)

        # Time advances by one step (10 minutes = 1/6 hour)
        state[6] = (state[6] + (10 / 60)) % 24

        return state

    def _apply_action(self, raw_state, action):
        """Apply actuator effects on top of natural drift."""
        state = raw_state.copy()
        for var_name, delta in ACTUATOR_EFFECTS[action].items():
            idx = PARAM_NAMES.index(var_name)
            state[idx] += delta
        return state

    def _clip_state(self, raw_state):
        """Keep all variables within simulation bounds."""
        state = raw_state.copy()
        for i, name in enumerate(PARAM_NAMES):
            lo, hi, _, _ = PARAMS[name]
            state[i] = np.clip(state[i], lo, hi)
        return state

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialise state with realistic starting values (random within ranges)
        raw = np.array([
            self.np_random.uniform(15, 30),   # temperature_c
            self.np_random.uniform(45, 65),   # humidity_pct
            0.0,                               # light_lux (starts at night)
            self.np_random.uniform(600, 1000), # co2_ppm
            self.np_random.uniform(35, 60),   # soil_moisture_pct
            self.np_random.uniform(100, 400), # voc_ppb
            self.np_random.uniform(0, 23),    # time_of_day
        ], dtype=np.float32)

        self._state = raw
        self._step_count = 0

        obs = self._raw_to_norm(self._state)
        info = {"raw_state": dict(zip(PARAM_NAMES, self._state.tolist()))}
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # 1. Natural drift
        new_state = self._natural_drift(self._state)

        # 2. Apply chosen action
        new_state = self._apply_action(new_state, action)

        # 3. Clip to simulation bounds
        new_state = self._clip_state(new_state)

        # 4. Compute reward
        reward = self._compute_reward(new_state, action)

        # 5. Advance
        self._state = new_state
        self._step_count += 1
        done = self._step_count >= self.max_steps

        obs = self._raw_to_norm(self._state)
        info = {
            "raw_state": dict(zip(PARAM_NAMES, self._state.tolist())),
            "action_name": ACTIONS[action],
            "step": self._step_count,
        }

        return obs, reward, done, False, info


# ─── Manual test ──────────────────────────────────────────────────────────────
# Run this file directly to verify the environment works before training.
# Expected output: 144 steps, rewards between roughly -6 and +8 per step.

if __name__ == "__main__":
    print("Testing GreenhouseEnv manually...")
    env = GreenhouseEnv()
    obs, info = env.reset(seed=42)

    print(f"Initial state:")
    for name, val in info["raw_state"].items():
        lo, hi, opt_lo, opt_hi = PARAMS[name]
        in_opt = opt_lo <= val <= opt_hi
        flag = "OK" if in_opt else "OUT"
        print(f"  {name:<22}: {val:7.2f}  [{flag}]  optimal [{opt_lo}, {opt_hi}]")

    total_reward = 0
    action_counts = {i: 0 for i in range(N_ACTIONS)}

    for step in range(144):
        action = env.action_space.sample()  # random agent
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        action_counts[action] += 1
        if done:
            break

    print(f"\nEpisode complete: 144 steps (24 simulated hours)")
    print(f"Total reward (random agent): {total_reward:.2f}")
    print(f"Mean reward per step:        {total_reward / 144:.3f}")
    print(f"\nAction distribution:")
    for i, count in action_counts.items():
        print(f"  {ACTIONS[i]:<15}: {count} times ({100*count/144:.1f}%)")

    print("\ngreehouse_env.py is working correctly.")
    print("Next step: open greenhouse_rl_training.ipynb in Google Colab to train the PPO agent.")
