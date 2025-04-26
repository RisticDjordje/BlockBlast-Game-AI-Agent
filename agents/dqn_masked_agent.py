import argparse
import os
import sys
import time
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from blockblast_game.game_env import BlockGameEnv
from agents.agent_visualizer import visualize_agent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


class MaskableDQN(DQN):
    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
        mask=None,
        **kwargs,
    ):
        # 1) Raw Q-values
        obs_tensor, _ = self.policy.obs_to_tensor(observation)
        q_values = self.q_net(obs_tensor).cpu().data.numpy()  # (batch_size, n_actions)

        # 2) Build masks from each sub-env
        masks = np.array([env.env.action_masks() for env in self.env.envs])

        # 3) Invalidate impossible actions
        q_values[~masks] = -1e8

        # 4) Greedy actions
        greedy = q_values.argmax(axis=1)

        if deterministic:
            # inference mode: unwrap to scalar if single-env
            if greedy.shape[0] == 1:
                return int(greedy[0]), None
            return greedy, None

        # 5) ε-greedy for training: always return an array
        batch_size, _ = q_values.shape
        actions = greedy.copy()
        for i in range(batch_size):
            if np.random.rand() < self.exploration_rate:
                valid = np.nonzero(masks[i])[0]
                actions[i] = np.random.choice(valid)

        # during training, ALWAYS return an array of shape (n_envs,)
        return actions, None


def make_env(rank: int, seed: int = 0):
    """
    Create one Monitor-wrapped BlockGameEnv.
    We'll vectorize with DummyVecEnv, then patch sample() on the VecEnv.
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


def train_masked_dqn(total_timesteps: int = 500_000, continue_training: bool = False):
    # 1) Build a single-env VecEnv
    env = DummyVecEnv([make_env(0)])

    # 2) Patch the VecEnv's action_space.sample() so warm-up only sees valid moves
    env.action_space.sample = lambda: int(
        np.random.choice(np.nonzero(env.envs[0].env.action_masks())[0])
    )

    # 3) Checkpoint callback
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=MODELS_DIR,
        name_prefix="masked_dqn_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # 4) Load or create the MaskableDQN
    final_zip = os.path.join(MODELS_DIR, "final_masked_dqn_model.zip")
    if continue_training and os.path.isfile(final_zip):
        print("[masked_dqn] Continuing from existing model")
        model = MaskableDQN.load(final_zip, env=env)
    else:
        model = MaskableDQN(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=LOGS_DIR,
            learning_rate=1e-4,
            buffer_size=1_000_000,
            learning_starts=20_000,
            batch_size=128,
            gamma=0.99,
            tau=0.005,
            target_update_interval=500,
            exploration_fraction=0.05,
            exploration_initial_eps=0.5,
            exploration_final_eps=0.03,
            train_freq=(4, "step"),
            gradient_steps=2,
        )

    # 5) Train
    print(f"[masked_dqn] Training for {total_timesteps} timesteps...")
    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)
    print(f"[masked_dqn] Done in {(time.time() - start):.1f}s")

    # 6) Save
    model.save(os.path.join(MODELS_DIR, "final_masked_dqn_model"))
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
    parser.add_argument("-t", "--timesteps", help="total_timesteps for training", dest="total_timesteps", type=int, default=20_000_000)
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
    
    # start to execute main process
    if do_train:
        train_masked_dqn(total_timesteps, continue_training)

    if do_visualize:
        # Build a render env and patch its sample() as well
        render_env = Monitor(BlockGameEnv(render_mode="human"))
        render_env.action_space.sample = lambda: int(
            np.random.choice(np.nonzero(render_env.env.action_masks())[0])
        )

        model_file = os.path.join(MODELS_DIR, "final_masked_dqn_model.zip")
        print(f"[masked_dqn] Loading model from {model_file}")
        loaded = MaskableDQN.load(model_file, env=render_env)
        visualize_agent(
            render_env,
            loaded,
            episodes=10,
            delay=0.2,
            use_masks=True,
            window_title="Maskable DQN Agent",
        )
