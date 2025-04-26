import argparse
import os
import sys
from typing import Callable, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from blockblast_game.game_env import BlockGameEnv
from agents.agent_visualizer import visualize_agent  # pragma: no cover

# Ensure project root is on the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Directories
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def make_env(rank: int, seed: int = 0) -> Callable[[], BlockGameEnv]:
    """Factory for SubprocVecEnv."""

    def _init():
        env = BlockGameEnv()
        env = Monitor(env)
        try:
            env.reset(seed=seed + rank)
        except TypeError:
            env.seed(seed + rank)
        return env

    return _init


def _standard_ppo(env, **kwargs) -> PPO:
    return PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=LOGS_DIR,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        **kwargs,
    )


def train_ppo(
    *,
    num_envs: int = 4,
    total_timesteps: int = 1_000_000,
    save_path: Optional[str] = None,
    continue_training: bool = False,
    pretrained_path: Optional[str] = None,
):
    save_dir = save_path or MODELS_DIR
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    if continue_training and pretrained_path and os.path.isfile(pretrained_path):
        print(f"[ppo] Continuing from {pretrained_path}")
        model = PPO.load(pretrained_path, env=env)
        model.tensorboard_log = LOGS_DIR
        reset_flag = False
    else:
        model = _standard_ppo(env)
        reset_flag = True

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=save_dir,
        name_prefix="ppo_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        reset_num_timesteps=reset_flag,
    )

    final_path = os.path.join(save_dir, "final_ppo_model")
    model.save(final_path)
    print(f"[ppo] Training done: {final_path}.zip")
    return model

def get_args():
    """
        arguments parser in main function
        python -m agents/dqn_*_agent.py -c -v -m -t 1200000
    """
    parser = argparse.ArgumentParser()
    parser.description='please enter optional parameters: train, visualize, continue, timestamp ...'
    # add_argument:
    # default: total_timesteps(20_000_000), do_train(true), do_visualize(true), continue_training(false)
    parser.add_argument("-t", "--timesteps", help="total_timesteps for training", dest="total_timesteps", type=int, default=50_000_000)
    parser.add_argument("-m", "--train", help="train mode", dest="do_train", action="store_false") 
    parser.add_argument("-v", "--visualize", help="visualize result", dest="do_visualize", action="store_false") 
    parser.add_argument("-c", "--continue", help="continue training or not", dest="continue_training", action="store_true") 
    parser.add_argument("-e", "--env", help="num of env", dest="num_envs", type=int, default=8)

    # parser result
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    """
        Main function for agent training and visualizing
    """
    # args parsing
    args = get_args()
    total_timesteps = args.total_timesteps
    continue_training = args.continue_training
    do_train = args.do_train
    do_visualize = args.do_visualize
    num_envs = args.num_envs

    print(f'[INFO] Args:\n\t{total_timesteps=}\n\t{continue_training=}\n\t{do_train=}\n\t{do_visualize=}\n\t{num_envs=}')
    # user confirmation
    user_confirm = input("[Note] Are you sure? (y/n) ")
    if user_confirm.lower() == "n":
        print("[WARNING] Please input the args again. Exiting !")
        sys.exit(1)

    # Configuration
    num_envs = 8
    total_timesteps = 50_000_000
    continue_training = False
    do_train = True
    do_visualize = True

    if do_train:
        pretrained = os.path.join(MODELS_DIR, "final_ppo_model.zip")
        train_ppo(
            num_envs=num_envs,
            total_timesteps=total_timesteps,
            continue_training=continue_training,
            pretrained_path=pretrained,
        )

    if do_visualize:
        render_env = BlockGameEnv(render_mode="human")
        render_env = Monitor(render_env)
        model_file = os.path.join(MODELS_DIR, "final_ppo_model.zip")
        print(f"[ppo] Loading model from {model_file}")
        agent = PPO.load(model_file, env=render_env)
        visualize_agent(render_env, agent, episodes=10, delay=0.2, use_masks=False)
