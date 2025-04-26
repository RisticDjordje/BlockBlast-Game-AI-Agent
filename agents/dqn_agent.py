import argparse
import sys
import time
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from agents.agent_visualizer import visualize_agent
from blockblast_game.game_env import BlockGameEnv

# Determine paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    Args:
        rank (int): Index of the subprocess
        seed (int): The initial seed for the environment

    Returns:
        The function to create the environment
    """

    def _init():
        env = BlockGameEnv()
        env = Monitor(env)
        try:
            env.reset(seed=seed + rank)
        except TypeError:
            env.seed(seed + rank)
        return env

    return _init


def train_dqn(total_timesteps=100_000, save_path=None, continue_training=False):
    """
    Train a DQN agent on the block game environment.

    Args:
        total_timesteps (int): Total timesteps for training
        save_path (str): Directory to save model checkpoints
        continue_training (bool): Whether to continue from saved model

    Returns:
        The trained DQN model
    """
    # Default to our MODELS_DIR if none passed
    save_dir = save_path or MODELS_DIR

    # Create training environment
    env = DummyVecEnv([make_env(0)])

    # Check for existing model
    final_zip = os.path.join(save_dir, "final_dqn_model.zip")
    if continue_training and os.path.isfile(final_zip):
        print("Loading existing model for continued training...")
        model = DQN.load(final_zip, env=env)
    else:
        model = DQN(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=LOGS_DIR,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=64,
            gamma=0.99,
            tau=0.005,
            target_update_interval=500,
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            train_freq=(4, "step"),
            gradient_steps=1,
        )

    # Checkpoint callback
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=save_dir,
        name_prefix="dqn_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    print(f"Starting DQN training for {total_timesteps} timesteps...")
    start = time.time()

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)

    # Save final model (without .zip suffix)
    final_model = os.path.join(save_dir, "final_dqn_model")
    model.save(final_model)

    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s")
    return model

def get_args():
    """
        arguments parser in main function
        python -m agents/*_agent.py -c -v -m -t 1200000
    """
    parser = argparse.ArgumentParser()
    parser.description='please enter optional parameters: train, visualize, continue, timestamp ...'
    # add_argument:
    # default: total_timesteps(20_000_000), do_train(true), do_visualize(true), continue_training(false)
    parser.add_argument("-t", "--timesteps", help="total_timesteps for training", dest="total_timesteps", type=int, default=50_000_000)
    parser.add_argument("-m", "--train", help="train mode", dest="do_train", action="store_false") 
    parser.add_argument("-v", "--visualize", help="visualize result", dest="do_visualize", action="store_false") 
    parser.add_argument("-c", "--continue", help="continue training or not", dest="continue_training", action="store_true") 

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

    print(f'[INFO] Args:\n\t{total_timesteps=}\n\t{continue_training=}\n\t{do_train=}\n\t{do_visualize=}')
    # user confirmation
    user_confirm = input("[Note] Are you sure? (y/n) ")
    if user_confirm.lower() == "n":
        print("[WARNING] Please input the args again. Exiting !")
        sys.exit(1)

    if do_train:
        trained = train_dqn(
            total_timesteps=total_timesteps,
            save_path=MODELS_DIR,
            continue_training=continue_training,
        )

    if do_visualize:
        # Create a fresh environment for rendering
        env = BlockGameEnv(render_mode="human")
        env = Monitor(env)
        model_file = os.path.join(MODELS_DIR, "final_dqn_model.zip")
        print(f"Loading model from {model_file}")
        loaded = DQN.load(model_file)
        visualize_agent(
            env, loaded, episodes=1, delay=1, use_masks=False, window_title="DQN Agent"
        )
