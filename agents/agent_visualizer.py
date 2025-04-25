"""
Utility for visualizing agents playing the block game.
Provides a consistent interface for visualizing different types of agents.
"""

import time
import numpy as np


def visualize_agent(
    env,
    agent,
    episodes: int = 5,
    delay: float = 0.2,
    use_masks: bool = False,
    window_title: str | None = None,
):
    """Visualize an agent playing the block game.

    Args:
        env: The environment to use.
        agent: The trained agent (any model with a ``predict`` method).
        episodes: Number of episodes to run.
        delay: Delay (seconds) between frames for visualization.
        use_masks: Whether to pass action‐mask information to the agent.
        window_title: Optional title for the pygame window.
    """

    # ------------------------------------------------------------------
    # Window title helpers
    # ------------------------------------------------------------------
    if window_title and hasattr(env, "set_window_title"):
        env.set_window_title(window_title)
    elif window_title and hasattr(env.unwrapped, "set_window_title"):
        env.unwrapped.set_window_title(window_title)
    elif (
        window_title
        and hasattr(env, "unwrapped")
        and hasattr(env.unwrapped, "renderer")
        and env.unwrapped.renderer
        and hasattr(env.unwrapped.renderer, "set_window_title")
    ):
        env.unwrapped.renderer.set_window_title(window_title)

    # ------------------------------------------------------------------
    # Tracking containers
    # ------------------------------------------------------------------
    episode_scores: list[int | float] = []
    episode_rewards: list[float] = []
    episode_steps: list[int] = []

    # ------------------------------------------------------------------
    # Run requested episodes
    # ------------------------------------------------------------------
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        episode_score = 0

        while not done and step < 1000:
            # --------------------------------------------
            #  LOG: per‑step header
            # --------------------------------------------
            print(f"\n--- Episode {episode + 1} · Step {step + 1} ---")

            # ----------------------------------------------------------
            #  Action selection (optionally with action‑masks)
            # ----------------------------------------------------------
            if use_masks:
                if hasattr(env.unwrapped, "action_masks"):
                    action_masks = env.unwrapped.action_masks()
                elif hasattr(env, "action_masks"):
                    action_masks = env.action_masks()
                else:
                    raise AttributeError("Environment doesn't support action masks")

                valid_count = np.sum(action_masks)
                total_count = len(action_masks)
                print(f"Valid actions: {valid_count}/{total_count}")

                action, _ = agent.predict(
                    obs, action_masks=action_masks, deterministic=True
                )
            else:
                action, _ = agent.predict(obs, deterministic=True)

            # ----------------------------------------------------------
            #  Decode action for human‑readable logging
            # ----------------------------------------------------------
            shape_idx = action // 64
            position = action % 64
            row = position // 8
            col = position % 8
            print(
                f"Selected action: {action} (Shape {shape_idx}, Row {row}, Col {col})"
            )

            # ----------------------------------------------------------
            #  Visual cue – agent is "thinking"
            # ----------------------------------------------------------
            renderer = None
            if hasattr(env, "renderer") and env.renderer:
                renderer = env.renderer
            elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "renderer"):
                renderer = env.unwrapped.renderer

            if renderer:
                renderer.set_agent_action(shape_idx, row, col)
                renderer.set_agent_thinking(True)

            env.render()
            time.sleep(delay / 2)

            # ----------------------------------------------------------
            #  Environment step
            # ----------------------------------------------------------
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            action_valid = info.get("action_valid", True)
            print(f"Action valid: {action_valid}, Reward: {reward:.2f}")
            print(f"Cumulative reward: {total_reward:.2f}")

            if "score" in info:
                episode_score = info["score"]

            if renderer:
                renderer.set_agent_thinking(False)

            env.render()
            time.sleep(delay / 2)

            done = terminated or truncated
            step += 1

        # ------------------------------------------------------------------
        # Episode finished – gather stats
        # ------------------------------------------------------------------
        episode_scores.append(episode_score)
        episode_rewards.append(total_reward)
        episode_steps.append(step)

        print(
            f"Episode {episode + 1} finished: Score = {episode_score}, "
            f"Reward = {total_reward:.2f}, Steps survived = {step}"
        )
        time.sleep(1)  # short pause before next episode

    # ----------------------------------------------------------------------
    #  Performance summary
    # ----------------------------------------------------------------------
    avg_score = sum(episode_scores) / len(episode_scores)
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_steps = sum(episode_steps) / len(episode_steps)
    max_steps = max(episode_steps)

    print("\n=== Performance Summary ===")
    print(f"Average Game Score : {avg_score:.2f}")
    print(f"Average RL Reward  : {avg_reward:.2f}")
    print(f"Average Steps      : {avg_steps:.2f}")
    print(f"Max Steps Survived : {max_steps}")
