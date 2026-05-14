# Greenhouse RL Controller — Progress Log

**Project:** AI-powered greenhouse environment controller using Reinforcement Learning
**Goal:** Train a PPO agent to automatically control greenhouse conditions (temperature,
humidity, light, CO2, soil moisture) by learning from trial and error, calibrated
against real sensor data, with a complete edge AI deployment pathway for ESP32.

---

## Phase 1: Project setup
**Status:** Complete
**Date:** May 2026

### What was done
- Identified project concept: two-layer approach. Layer 1 is a browser simulation
  for non-technical audiences. Layer 2 is a Python PPO implementation in Colab.
- Searched for and identified datasets from Kaggle and IEEE DataPort.
- Downloaded 4 datasets from Kaggle (Marcel Boonman greenhouse sensor, Plant Growth
  Classification, Greenhouse Plant Growth Metrics, IoT Telemetry).
- Created folder structure on local machine.

### Files produced
- data/greenhouse_sensor_10min.csv
- data/plant_growth_conditions.csv
- data/greenhouse_growth_metrics.csv (later removed, see Phase 2)
- data/iot_telemetry.csv (later removed, see Phase 2)

### Tools used
- Kaggle (free account), VS Code

---

## Phase 2: Data analysis
**Status:** Complete
**Date:** May 2026

### What was done
- Inspected all 4 datasets for column names, row counts, data types, and value ranges.
- Identified which datasets are useful and which are not.
- Extracted realistic parameter ranges for every greenhouse variable from the real data.
- Saved parameter ranges as a JSON reference file used by all future phases.
- Renamed dataset files to clear, readable names.

### Key decisions made

**Removed: greenhouse_growth_metrics.csv (30,000 rows)**
Reason: All column names were cryptic abbreviations (ACHP, PHR, ALAP, etc.) with no
data dictionary provided. Without knowing what the columns mean, the data cannot be
used reliably. Removed rather than risk building the environment on misunderstood data.

**Removed: iot_telemetry.csv (405,184 rows)**
Reason: This dataset measures CO (carbon monoxide), not CO2. These are completely
different gases. The light column is boolean (on/off only), not a lux value. Once
the proper greenhouse sensor dataset arrived from Marcel Boonman, this dataset became
entirely redundant.

**Kept: greenhouse_sensor_10min.csv (17,562 rows)**
Real greenhouse sensor readings at 10-minute intervals. Contains indoor temperature,
humidity, light in lux, CO2 in ppm, and VOC in ppb. Primary calibration dataset.

**Kept: plant_growth_conditions.csv (193 rows)**
Small but clean dataset linking temperature, humidity, and sunlight hours to binary
plant growth outcomes. Used to validate the reward function optimal ranges.

### Parameter reference table (saved to data/parameter_ranges.json)

| Variable      | Simulation range  | Optimal zone (reward) | Source                           |
|---------------|-------------------|-----------------------|----------------------------------|
| Temperature   | 5 to 45°C         | 18 to 26°C            | greenhouse_sensor + plant_growth |
| Humidity      | 20 to 85%         | 50 to 70%             | greenhouse_sensor + plant_growth |
| Light         | 0 to 35,000 lux   | 3,000 to 10,000 lux   | greenhouse_sensor                |
| CO2           | 400 to 2,000 ppm  | 800 to 1,500 ppm      | greenhouse_sensor + literature   |
| Soil moisture | 10 to 90%         | 40 to 70%             | plant_growth + literature        |
| VOC           | 0 to 2,000 ppb    | 0 to 500 ppb          | greenhouse_sensor                |

Note: Soil moisture optimal range comes from horticultural literature as the
greenhouse sensor dataset does not include soil moisture readings.

### Files produced
- data/greenhouse_sensor_10min.csv (cleaned and renamed)
- data/plant_growth_conditions.csv (renamed)
- data/parameter_ranges.json

### Files removed
- data/greenhouse_growth_metrics.csv
- data/iot_telemetry.csv

### Tools used
- Python (pandas, numpy), Claude for data analysis

---

## Phase 3: Environment design
**Status:** Complete
**Date:** May 2026

### What was done
- Designed and implemented a custom Gymnasium environment simulating the greenhouse.
- Defined the state space, action space, reward function, and actuator physics.
- Tested the environment locally in VS Code with a random agent.
- Confirmed the environment runs correctly and produces expected output.

### Design decisions explained

**State space: 7 variables**
These are the 7 values the AI agent can see at each step, equivalent to what
physical sensors would read in a real greenhouse:
1. temperature_c: indoor air temperature in Celsius
2. humidity_pct: relative humidity percentage
3. light_lux: light intensity in lux
4. co2_ppm: CO2 concentration in parts per million
5. soil_moisture_pct: soil moisture percentage
6. voc_ppb: volatile organic compounds in parts per billion
7. time_of_day: hour of the day (0 to 23), so the agent knows day from night

All 7 variables are normalised to [0, 1] before being passed to the agent.
This is standard practice in RL: it prevents one variable dominating because
it has a larger numerical range than others.

**Action space: 6 discrete actions**
These are the 6 things the AI agent can do at each step, equivalent to switching
relays on an ESP32 controller board:
- 0: Do nothing
- 1: Open vents (cools air, reduces humidity, removes CO2 and VOC)
- 2: Water plants (raises soil moisture and humidity)
- 3: Turn on grow lights (raises light intensity, slightly raises temperature)
- 4: Turn on heater (raises temperature, slightly reduces humidity)
- 5: Inject CO2 (raises CO2 concentration)

**Reward function**
At each step the agent receives a score:
- +1.0 for each variable currently inside its optimal range (max +6.0)
- Scaled penalty (down to -1.0) for each variable outside its optimal range
- +2.0 bonus if every single variable is simultaneously in optimal range
- Energy cost deducted per action (heater: -0.8, vents: -0.1, others between)
- Maximum possible reward per step: +8.0

**Episode length: 144 steps**
One episode = 144 steps = 24 simulated hours at 10-minute intervals.
This matches the 10-minute interval of the Marcel Boonman dataset exactly.

**Actuator physics**
Each action changes variables by realistic amounts per step calibrated against
the rate of change seen in the real sensor data. For example, opening vents
reduces temperature by 1.5°C and humidity by 3% per step.

### Test results (random agent baseline)
- Total reward per episode: varied between -62 and +77 across runs (seed dependent)
- This low baseline is the benchmark the trained PPO agent improves against

### Files produced
- code/greenhouse_env.py

### Tools used
- Python, gymnasium, numpy, VS Code

---

## Phase 4: Browser simulation and Q-Learning baseline
**Status:** Complete
**Date:** May 2026

### What was done
- Built a standalone browser simulation running Q-Learning in JavaScript.
- Built a Python Q-Learning agent for the baseline comparison.
- Generated learning curve chart.

### Part 1: Python Q-Learning agent (q_agent.py)

**Q-table design**
State: ternary encoding — too low (0), optimal (1), too high (2) — per variable.
3^6 = 729 possible states. Action space: 6. Q-table: 729 × 6 = 4,374 values.
Ternary encoding allows the agent to distinguish temperature too HOT (open vents)
from temperature too COLD (heater on). Binary encoding collapsed both into one state.

**Hyperparameters**
- Learning rate: 0.1
- Discount factor: 0.95
- Epsilon: 1.0 decaying to 0.05 over 79 episodes at 0.012 per episode
- Total training: 500 episodes

**Results**
- First 10 episodes average: 170.5
- Last 10 episodes average: 241.2
- Improvement: +70.7 confirmed

**Limitation found (important for project narrative)**
Initial limitation found and fixed: binary state encoding (in/out of range) meant
the agent could not distinguish too-high from too-low for any variable. Fixed by
switching to ternary encoding (3^6 = 729 states). After the fix the agent
correctly learns: temp too hot → vents, temp too cold → heater, dark → lights,
dry → water, low CO2 → inject. Q-learning still has a performance ceiling vs PPO
due to discrete state approximation, but decisions are now qualitatively correct.

### Part 2: Browser simulation (simulation.html)

Single standalone HTML file, 37 KB. Opens in any browser by double-clicking.
No server, no installation, no login required. Implements Q-learning in JavaScript.

Features:
- Animated greenhouse cross-section with day/night cycle
- 6 live sensor gauges with optimal zone highlighting
- Episode counter and real-time learning curve chart
- Speed slider (1x to 100x)
- Downloadable episode data CSV export

Audience: non-technical viewers, demo at interviews, LinkedIn.

### Files produced
- code/q_agent.py
- code/simulation.html
- results/learning_curve.png

---

## Phase 5: PPO training in Google Colab
**Status:** Complete
**Date:** May 2026

### What was done
- Trained PPO agent using Stable Baselines 3 in Google Colab.
- Evaluated against Q-Learning and random baselines over 100 episodes each.
- Exported all results as CSV files and PNG charts.

### Training configuration

| Parameter      | Value          |
|----------------|----------------|
| Policy         | MlpPolicy      |
| Architecture   | 7 → 64 → 64 → 6, tanh activation |
| Total params   | 9,799 (actor + critic) |
| Learning rate  | 3e-4           |
| n_steps        | 2,048          |
| batch_size     | 64             |
| clip_range     | 0.2            |
| Timesteps      | 200,000        |
| Episodes       | 1,393          |
| Training time  | 5m 47s on Colab free CPU |

### Training progression (ep_rew_mean from SB3 logs)

| Milestone      | Mean reward |
|----------------|-------------|
| Start          | 174         |
| Episode ~85    | 412         |
| Episode ~185   | 800         |
| Final          | 865         |
| First 20 avg   | 177.1       |
| Last 20 avg    | 831.5       |
| Improvement    | +654.4      |

### Evaluation results (100 episodes each, deterministic policy)

| Agent         | Mean reward | Std   | Min     | Max      |
|---------------|-------------|-------|---------|----------|
| Random        | 173.9       | 115.4 | -63.91  | 405.00   |
| Q-Learning    | 311.9       | 111.0 | -328.59 | 548.90   |
| PPO (trained) | 817.9       | 127.5 | 530.86  | 1,043.92 |

PPO improvement over Q-Learning: +506.0 (162% better)
PPO improvement over Random: +644.0 (370% better)

Note: PPO minimum reward (530) exceeds Q-Learning mean (312). The worst PPO
episode outperforms an average Q-Learning episode.

### Edge AI feasibility check
- Model size float32: 38.3 KB (full model)
- Model size int8 quantised: 9.6 KB
- ESP32 SRAM available: 520 KB
- Fits on ESP32: Yes

### Known issue fixed before GitHub upload
Original notebook cell 9 had a broken matplotlib color string that caused a
ValueError. Fixed by replacing invalid CSS string with `color='#52b788', alpha=0.3`.

### Notes
- DeprecationWarning from matplotlib boxplot labels parameter is harmless.
- Warning about gym vs gymnasium is harmless, SB3 handles this internally.
- Q-Learning was retrained fresh in Colab for a fair comparison on same seeds.

### Files produced
- results/greenhouse_ppo_model.zip (trained model, 145 KB)
- results/ppo_learning_curve.png
- results/agent_comparison.png
- results/ppo_training_rewards.csv (1,393 episode rewards)
- results/agent_comparison_results.csv (100 episodes per agent)
- results/results_summary.csv

---

## Phase 6: Edge AI framing
**Status:** Complete (deployment design and weight extraction done; hardware build deferred)
**Date:** May 2026

### What was done

**Weight extraction (Colab)**
Ran the weight extraction cell (cell 16) in the Colab notebook after Phase 5 training. Extracted actor network
only. The critic network inside PPO is used only during training and is not needed
for inference on hardware.

| Layer | Shape    | Values | Description              |
|-------|----------|--------|--------------------------|
| W1    | (64, 7)  | 448    | Input → hidden 1         |
| b1    | (64,)    | 64     | Hidden 1 bias            |
| W2    | (64, 64) | 4,096  | Hidden 1 → hidden 2      |
| b2    | (64,)    | 64     | Hidden 2 bias            |
| W3    | (6, 64)  | 384    | Hidden 2 → output        |
| b3    | (6,)     | 6      | Output bias              |
| Total |          | 5,062  | Actor network only       |

Note: Phase 5 reported 9,799 (full model including critic). Phase 6 uses 5,062
(actor only). Both are correct for their respective contexts.

Verification: Python forward pass using extracted weights matched SB3 model
prediction exactly. Weights correctly extracted confirmed.

File: nn_weights.h, 57.6 KB float32
Int8 quantised estimate: 14.4 KB
ESP32 SRAM: 520 KB. Model fits with room to spare.

**ESP32 firmware (code complete, hardware build deferred)**
Complete Arduino firmware written: esp32_greenhouse_controller.ino
Implements the same control loop as the simulation:
1. Read sensors via GPIO (DHT22, soil sensor, LDR, MQ135, RTC)
2. Normalise to [0, 1] using parameter_ranges.json values
3. Run neural network forward pass (3 matrix multiplications + tanh)
4. argmax of 6 output logits gives action index
5. Fire relay GPIO for the chosen actuator
6. Wait 10 minutes, repeat (144 cycles per day)

Estimated inference time: under 1 ms at ESP32's 240 MHz.

Hardware build is deferred until a new laptop is available for Arduino IDE
installation and USB flashing. All code is complete and ready to flash.
This does not weaken the project: simulation validates the control logic
and the deployment design documents hardware feasibility.

**System architecture diagram**
Designed and saved as results/system_architecture.svg. Shows the full pipeline:
sensors → ESP32 (normalise → neural network → argmax) → relay board → actuators.
Opens in any browser. Vector format, stays sharp at any size.

### Why the hardware deferral is not a weakness
Simulation-validated RL with a documented deployment pathway is a complete and
credible embedded AI contribution. The project's claim is that the PPO policy
learns to control greenhouse conditions effectively, proven by the Phase 5 results.
Phase 6 demonstrates hardware feasibility, not execution. This framing is
standard in academic embedded AI work.

### Hardware bill of materials (for future build, estimated)

| Component              | Purpose               | Approx. cost |
|------------------------|-----------------------|-------------|
| ESP32 Dev Module       | Microcontroller       | £6           |
| DHT22                  | Temperature, humidity | £4           |
| Capacitive soil sensor | Soil moisture         | £3           |
| LDR + resistor         | Light intensity proxy | £1           |
| MQ135 gas sensor       | CO2, VOC proxy        | £5           |
| 5-channel relay board  | Actuator control      | £6           |
| DS3231 RTC module      | Accurate time of day  | £3           |
| Wiring and breadboard  | Prototyping           | £4           |
| Total estimate         |                       | ~£32         |

### Files produced
- code/esp32_greenhouse_controller/esp32_greenhouse_controller.ino
- code/esp32_greenhouse_controller/nn_weights.h (actual trained weights, 57.6 KB)
- results/system_architecture.svg

---

## Phase 7: Demonstrate (GitHub + LinkedIn)
**Status:** In progress
**Date:** May 2026

### What was done
- Created README.md for GitHub repository.
- Fixed notebook cell 9 bug before upload.
- Created .gitignore to exclude __pycache__ and compiled files.
- Updated this progress log to reflect all phases accurately.

### Remaining steps
1. Add LinkedIn URL and GitHub username to the bottom of README.md
2. Create GitHub repository named greenhouse-rl-controller
3. Delete code/__pycache__/ folder before uploading
4. Upload all files to GitHub
5. Write LinkedIn post using agent_comparison.png and ppo_learning_curve.png

---

## Complete folder structure

```
greenhouse-rl/
    data/
        greenhouse_sensor_10min.csv       Phase 2: primary calibration dataset
        plant_growth_conditions.csv       Phase 2: reward function calibration
        parameter_ranges.json             Phase 2: single source of truth
    code/
        greenhouse_env.py                 Phase 3: custom Gymnasium environment
        q_agent.py                        Phase 4: Q-Learning baseline agent
        simulation.html                   Phase 4: browser demo, no install needed
        esp32_greenhouse_controller/
            esp32_greenhouse_controller.ino   Phase 6: Arduino firmware
            nn_weights.h                      Phase 6: actual trained weights
    results/
        learning_curve.png                Phase 4: Q-Learning curve
        greenhouse_ppo_model.zip          Phase 5: trained PPO model
        ppo_learning_curve.png            Phase 5: PPO training curve
        agent_comparison.png              Phase 5: key comparison chart
        ppo_training_rewards.csv          Phase 5: episode-by-episode data
        agent_comparison_results.csv      Phase 5: evaluation data
        results_summary.csv               Phase 5: mean/std/min/max per agent
        system_architecture.svg           Phase 6: hardware deployment diagram
    README.md                             Phase 7: GitHub portfolio page
    progress_log.md                       All phases: full documentation
    greenhouse_rl_training.ipynb          Phase 5 + 6: training notebook (cell 16 extracts weights)
    .gitignore                            Phase 7: excludes __pycache__ etc.
```

---

## Results summary

| Phase | Key output              | Key number                                       |
|-------|-------------------------|--------------------------------------------------|
| 2     | parameter_ranges.json   | 6 variables from 17,562 real sensor readings     |
| 3     | greenhouse_env.py       | 7 state variables, 6 actions, 144 steps/episode  |
| 4     | simulation.html         | Q-Learning: 170.5 → 241.2 (+70.7)               |
| 5     | PPO training            | PPO 818 vs Q-Learning 312 vs Random 174          |
| 6     | nn_weights.h + .ino     | 5,062 actor params, 14.4 KB, fits ESP32 520 KB   |
