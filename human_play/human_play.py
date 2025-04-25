import json, os, time, sys
from typing import List, Any

import pygame
import numpy as np
from blockblast_game.game_env import BlockGameEnv

LOG_FILE = "human_play/human_play_log.json"
FPS = 30


# ---------- helpers ---------------------------------------------------------
def _to_python(obj: Any):
    """Recursively turn ndarray ⇢ list so json.dump() is happy."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


def _load_existing_log() -> List[dict]:
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("[warning] log corrupted – starting fresh")
    return []


def _save_log(data: List[dict]):
    tmp = LOG_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, LOG_FILE)  # atomic write


# ---------- main ------------------------------------------------------------
def main():
    env = BlockGameEnv(render_mode="human")
    obs, _ = env.reset()
    log_data = _load_existing_log()
    new_entries: List[dict] = []

    clock = pygame.time.Clock()
    try:
        running = True
        while running:
            env.render()
            act = env.renderer.process_human_events()

            if act == "QUIT":
                running = False  # graceful window-close
                continue
            if act == "RESET":
                obs, _ = env.reset()
                continue
            if act:
                shape_idx, row, col = act
                flat = shape_idx * 64 + row * 8 + col

                state_before = obs
                obs, rew, term, trunc, info = env.step(flat)

                new_entries.append(
                    {
                        "state": _to_python(state_before),
                        "action": flat,
                        "reward": float(rew),
                        "next_state": _to_python(obs),
                        "info": _to_python(info),
                        "timestamp": time.time(),
                    }
                )

                if term:
                    obs, _ = env.reset()

            clock.tick(FPS)

    except SystemExit:
        # sys.exit() raised inside game_renderer. We'll still write the file.
        pass
    finally:
        if new_entries:
            log_data.extend(new_entries)
            _save_log(log_data)
            print(
                f"[play] appended {len(new_entries)} steps → {LOG_FILE} "
                f"(total: {len(log_data)})"
            )
        else:
            print("[play] no actions recorded – nothing saved")


if __name__ == "__main__":
    main()
