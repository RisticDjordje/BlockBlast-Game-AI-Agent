import os
import sys
import logging
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, DQN

try:
    from sb3_contrib import MaskablePPO

    maskable_ppo_available = True
except ImportError:
    maskable_ppo_available = False

# Monkey-patch numpy for SB3 compatibility
import numpy.core.numeric

sys.modules["numpy._core.numeric"] = numpy.core.numeric

from blockblast_game.game_env import BlockGameEnv

# Define directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "agents", "models"))


# Configure logging
def configure_logging(log_file: str = None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


# Helper to construct model paths
def model_path(name: str) -> str:
    return os.path.join(MODELS_DIR, name)


# Run episodes for one agent and log per-episode metrics
def run_agent(
    env, agent, agent_name: str, episodes=300, max_steps=1000, use_masks=False
):
    logger = logging.getLogger()
    scores, rewards, steps, valid_moves, invalid_attempts = [], [], [], [], []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = total_steps = valid = invalid = 0
        score = 0

        while not done and total_steps < max_steps:
            if isinstance(agent, str) and agent == "random":
                valid_actions = env.get_valid_actions()
                action = np.random.choice(valid_actions)
            else:
                if use_masks and hasattr(env, "action_masks"):
                    action, _ = agent.predict(obs, action_masks=env.action_masks())
                else:
                    action, _ = agent.predict(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            score = info.get("score", score)
            done = terminated or truncated

            if info.get("valid_placement", False):
                valid += 1
            else:
                invalid += 1

        # Record metrics
        scores.append(score)
        rewards.append(total_reward)
        steps.append(total_steps)
        valid_moves.append(valid)
        invalid_attempts.append(invalid)

        # Log per-episode summary
        logger.info(
            f"{agent_name} - Episode {ep}/{episodes} | score={score} | reward={total_reward} | "
            f"steps={total_steps} | valid_moves={valid} | invalid_attempts={invalid}"
        )

    return scores, rewards, steps, valid_moves, invalid_attempts


def main():
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Initialize logging
    log_path = os.path.join(RESULTS_DIR, "simulation.log")
    configure_logging(log_file=log_path)
    logging.info(f"Logs will be saved to {log_path}")

    # Create environment
    env = BlockGameEnv(render_mode=None)

    # Load agents
    agents = {"Random": "random"}

    ppo_file = model_path("final_ppo_model.zip")
    if os.path.isfile(ppo_file):
        agents["PPO"] = PPO.load(ppo_file)
    else:
        logging.warning(f"PPO model not found at {ppo_file}")

    dqn_file = model_path("final_dqn_model.zip")
    if os.path.isfile(dqn_file):
        agents["DQN"] = DQN.load(dqn_file)
    else:
        logging.warning(f"DQN model not found at {dqn_file}")

    mp_file = model_path("final_masked_ppo_model.zip")
    if maskable_ppo_available and os.path.isfile(mp_file):
        agents["Masked PPO"] = MaskablePPO.load(mp_file)
    elif maskable_ppo_available:
        logging.warning(f"Masked PPO model not found at {mp_file}")
    else:
        logging.warning("sb3_contrib.MaskablePPO not available; skipping Masked PPO")

    # Run experiments
    all_results = {}
    for name, agent in agents.items():
        logging.info(f"Starting runs for {name}")
        use_masks = name == "Masked PPO"
        results = run_agent(
            env,
            agent,
            agent_name=name,
            episodes=500,
            max_steps=1000,
            use_masks=use_masks,
        )
        all_results[name] = {
            "scores": results[0],
            "rewards": results[1],
            "steps": results[2],
            "valid_moves": results[3],
            "invalid_attempts": results[4],
        }

    # Flatten and save to CSV
    rows = []
    for agent_name, data in all_results.items():
        for ep_idx, (sc, rw, st, vm, im) in enumerate(
            zip(
                data["scores"],
                data["rewards"],
                data["steps"],
                data["valid_moves"],
                data["invalid_attempts"],
            ),
            start=1,
        ):
            rows.append(
                {
                    "agent": agent_name,
                    "episode": ep_idx,
                    "score": sc,
                    "reward": rw,
                    "steps": st,
                    "valid_moves": vm,
                    "invalid_attempts": im,
                }
            )
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved all episode results to {csv_path}")


if __name__ == "__main__":
    main()
