"""
q_agent.py
==========
Step 1: Q-learning agent for the greenhouse environment.

What this file does:
    Trains a Q-learning agent to control the greenhouse.
    Prints episode rewards to the terminal so you can watch
    the agent improving over 200 episodes.
    Saves a learning curve chart as results/learning_curve.png.

How to run:
    python code/q_agent.py

What you should see:
    Early episodes: reward around -50 to +100 (random guessing)
    Middle episodes: reward climbing toward +200 to +400
    Late episodes: reward stabilising above +400 (agent learned)

How Q-learning works (plain explanation):
    Imagine a table. Each ROW is a situation the greenhouse can be in.
    Each COLUMN is an action (vent, water, lights, etc.).
    Each CELL holds a score: how good is this action in this situation?

    At the start every cell is 0. The agent picks random actions
    (exploration). After each action it updates the cell:
        new score = old score + learning_rate * (actual reward + expected future - old score)

    Over time the good actions get higher scores. The agent starts
    preferring high-scored actions over random ones (exploitation).
    This is the entire Q-learning algorithm.

Q-table design:
    State: for each of 6 variables, is it IN the optimal range or OUT?
    That gives 2^6 = 64 possible states.
    Actions: 6 discrete actions.
    Q-table size: 64 rows x 6 columns = 384 numbers.
"""

import numpy as np
import os
import sys

# Make sure greenhouse_env.py is findable whether you run from
# the project root or the code/ folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from greenhouse_env import GreenhouseEnv, PARAMS, PARAM_NAMES, N_ACTIONS, ACTIONS

# ─── Q-learning hyperparameters ───────────────────────────────────────────────
# These three numbers control how the agent learns.

LEARNING_RATE = 0.1
# How much the agent updates its beliefs after each step.
# 0.1 = take 10% of new information each time.
# Too high (>0.5): agent forgets old knowledge, unstable.
# Too low (<0.01): agent learns very slowly.

DISCOUNT = 0.95
# How much the agent values future rewards vs immediate ones.
# 0.95 = future rewards are worth 95% as much as immediate ones.
# 1.0 = agent cares about all future rewards equally.
# 0.0 = agent is completely short-sighted.

EPSILON_START = 1.0     # Start: explore completely randomly
EPSILON_END   = 0.05    # End: explore only 5% of the time
EPSILON_DECAY = 0.012   # Reduce epsilon by this much per episode
                        # Reaches 0.05 after about 79 episodes

N_EPISODES = 200        # Total training episodes
PRINT_EVERY = 10        # Print progress every N episodes


# ─── State encoding ───────────────────────────────────────────────────────────

def get_state_key(raw_state: dict) -> int:
    """
    Convert the raw state dictionary into a single integer 0-63.

    For each of the 6 sensor variables, ask: is it in the optimal range?
    Yes = 1, No = 0. This gives a 6-bit binary number.

    Example:
        temp IN range, humidity OUT, light IN, co2 IN, soil OUT, voc IN
        → bits: 1 0 1 1 0 1 = binary 101101 = decimal 45

    Why only 6 variables, not 7?
        time_of_day is not controlled by the agent. It advances on its
        own. Including it would just add noise to the state encoding.
    """
    key = 0
    controllable = PARAM_NAMES[:6]  # exclude time_of_day
    for i, name in enumerate(controllable):
        p = PARAMS[name]          # p = [sim_min, sim_max, opt_min, opt_max]
        value = raw_state[name]
        opt_min = p[2]
        opt_max = p[3]
        if opt_min <= value <= opt_max:
            key |= (1 << i)
    return key  # integer 0 to 63


# ─── Q-table ──────────────────────────────────────────────────────────────────

N_STATES = 64  # 2^6 binary states

def create_q_table() -> np.ndarray:
    """Create an empty Q-table of shape (64 states, 6 actions)."""
    return np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

def choose_action(q_table: np.ndarray, state_key: int, epsilon: float) -> int:
    """
    Epsilon-greedy action selection.

    With probability epsilon: pick a random action (explore).
    Otherwise: pick the action with the highest Q-value (exploit).

    Early training (epsilon=1.0): 100% random.
    Late training (epsilon=0.05): 95% exploit, 5% explore.
    """
    if np.random.random() < epsilon:
        return np.random.randint(N_ACTIONS)   # random action
    return int(np.argmax(q_table[state_key])) # best known action

def update_q_table(
    q_table:   np.ndarray,
    state:     int,
    action:    int,
    reward:    float,
    new_state: int
) -> None:
    """
    Update one cell in the Q-table after taking an action.

    Formula (Bellman equation):
        Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_Q(s') - Q(s, a)]

    In plain English:
        New score = old score + learning_rate * (what actually happened - what we expected)

    This is called the Bellman equation. It is the core of Q-learning.
    """
    current_q  = q_table[state, action]
    best_next  = np.max(q_table[new_state])
    target     = reward + DISCOUNT * best_next
    q_table[state, action] = current_q + LEARNING_RATE * (target - current_q)


# ─── Training loop ────────────────────────────────────────────────────────────

def train():
    env     = GreenhouseEnv()
    q_table = create_q_table()
    epsilon = EPSILON_START

    episode_rewards = []  # total reward per episode
    episode_optimal = []  # average variables in optimal range per step

    print("=" * 60)
    print("Greenhouse Q-Learning Agent — Training")
    print("=" * 60)
    print(f"Episodes:      {N_EPISODES}")
    print(f"Steps/episode: 144  (24 simulated hours)")
    print(f"Q-table size:  {N_STATES} states × {N_ACTIONS} actions")
    print(f"Epsilon decay: {EPSILON_START} → {EPSILON_END} over ~79 episodes")
    print("=" * 60)
    print()

    for episode in range(N_EPISODES):

        # Reset environment for new episode
        _, info = env.reset(seed=None)
        raw_state = info["raw_state"]
        state_key = get_state_key(raw_state)

        total_reward = 0.0
        total_optimal = 0
        action_counts = {i: 0 for i in range(N_ACTIONS)}

        for step in range(144):

            # 1. Choose action
            action = choose_action(q_table, state_key, epsilon)
            action_counts[action] += 1

            # 2. Take action in environment
            _, reward, done, _, info = env.step(action)
            new_raw_state = info["raw_state"]
            new_state_key = get_state_key(new_raw_state)

            # 3. Update Q-table
            update_q_table(q_table, state_key, action, reward, new_state_key)

            # 4. Track progress
            total_reward += reward
            # Count how many variables are in optimal range this step
            in_opt = sum(
                1 for name in PARAM_NAMES[:6]
                if PARAMS[name][2] <= new_raw_state[name] <= PARAMS[name][3]
            )
            total_optimal += in_opt

            # 5. Move to next state
            state_key = new_state_key
            raw_state = new_raw_state

            if done:
                break

        # End of episode
        episode_rewards.append(total_reward)
        avg_optimal = total_optimal / 144

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon - EPSILON_DECAY)

        # Print progress every N episodes
        if (episode + 1) % PRINT_EVERY == 0 or episode == 0:

            # Rolling average of last 10 episodes
            recent = episode_rewards[-10:]
            avg_reward = np.mean(recent)

            # Most used action this episode
            top_action = max(action_counts, key=action_counts.get)

            # Mode label
            if epsilon > 0.7:
                mode = "Exploring"
            elif epsilon > 0.3:
                mode = "Learning "
            elif epsilon > 0.1:
                mode = "Exploiting"
            else:
                mode = "Trained   "

            print(
                f"Ep {episode+1:>3} | "
                f"Reward: {total_reward:>8.1f} | "
                f"10-ep avg: {avg_reward:>8.1f} | "
                f"Optimal/step: {avg_optimal:.2f}/6 | "
                f"ε={epsilon:.2f} | {mode}"
            )

    print()
    print("=" * 60)
    print("Training complete.")
    print(f"First 10 episodes avg:  {np.mean(episode_rewards[:10]):>8.1f}")
    print(f"Last  10 episodes avg:  {np.mean(episode_rewards[-10:]):>8.1f}")
    improvement = np.mean(episode_rewards[-10:]) - np.mean(episode_rewards[:10])
    print(f"Improvement:            {improvement:>+8.1f}  {'✓ Agent learned' if improvement > 50 else '? Check parameters'}")
    print("=" * 60)

    return episode_rewards, q_table


# ─── Chart ────────────────────────────────────────────────────────────────────

def save_chart(episode_rewards: list):
    """
    Save the learning curve as results/learning_curve.png.
    Shows episode reward and 10-episode rolling average.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        print("Chart not saved. All other results are unaffected.")
        return

    # Create results folder
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Compute rolling average
    rolling = [
        np.mean(episode_rewards[max(0, i-9):i+1])
        for i in range(len(episode_rewards))
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = range(1, len(episode_rewards) + 1)

    ax.plot(episodes, episode_rewards,
            color='#B4B2A9', linewidth=1, alpha=0.8, label='Episode reward')
    ax.plot(episodes, rolling,
            color='#534AB7', linewidth=2.5, label='10-episode average')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Total reward', fontsize=11)
    ax.set_title('Greenhouse Q-Learning: Learning Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    # Annotate improvement
    first10 = np.mean(episode_rewards[:10])
    last10  = np.mean(episode_rewards[-10:])
    ax.annotate(f'Start avg: {first10:.0f}',
                xy=(10, first10), fontsize=9, color='#888780')
    ax.annotate(f'Final avg: {last10:.0f}',
                xy=(len(episode_rewards)-10, last10), fontsize=9, color='#534AB7')

    chart_path = os.path.join(results_dir, 'learning_curve.png')
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nChart saved: {chart_path}")
    print("Open results/learning_curve.png to see the learning curve.")


# ─── Q-table inspection ───────────────────────────────────────────────────────

def inspect_policy(q_table: np.ndarray):
    """
    Print what the trained agent has learned.
    Shows the best action for a few key situations.
    """
    print()
    print("=" * 60)
    print("What the agent learned (sample of trained policy):")
    print("=" * 60)

    # A few representative states
    situations = {
        "All variables in optimal range":     0b111111,  # all 6 bits set = 63
        "Temperature too high (OUT of range)":0b111110,  # temp bit = 0
        "Soil moisture too low":              0b101111,  # soil bit = 0
        "Light too low":                      0b111011,  # light bit = 0 (bit 2)
        "CO2 too low":                        0b110111,  # co2 bit = 0
        "Nothing in optimal range":           0b000000,  # all out = 0
    }

    for label, state_key in situations.items():
        best_action  = int(np.argmax(q_table[state_key]))
        best_q_value = q_table[state_key, best_action]
        action_name  = ACTIONS[best_action]
        print(f"  {label}")
        print(f"    → Best action: {action_name} (Q-value: {best_q_value:.2f})")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(None)  # different result each run

    # Train
    episode_rewards, q_table = train()

    # Show what was learned
    inspect_policy(q_table)

    # Save chart
    save_chart(episode_rewards)

    print()
    print("Next step: open greenhouse_rl_training.ipynb in Google Colab")
    print("to train a more powerful PPO agent (Phase 5).")
