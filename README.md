# Block Blast Game + Reinforcement Learning AI Agent

Block Blast is a Tetris-inspired puzzle game played on an 8×8 grid. At each turn, the player (or agent) is presented with three randomly generated block shapes and must choose one to place anywhere on the board. Whenever an entire row or column is filled, it clears and awards points; clearing multiple lines in succession activates a combo multiplier for even higher scores.

Under the hood, the game engine guarantees that at least one of the three available shapes can always be placed, ensuring no unwinnable states arise prematurely. A Pygame-based interface lets humans play and test strategies interactively, while a custom OpenAI Gym environment exposes the game’s state, action space, and reward function for Reinforcement Learning.

This repository implements several RL agents—including a random baseline, Deep Q-Network (DQN), Proximal Policy Optimization (PPO), and a masked-action PPO variant—to train and evaluate performance. Comprehensive scripts are provided to train agents, log metrics, visualize results, and compare different approaches under consistent conditions.

<p align="center">
  <img src="blockblast_game/Assets/demo.png" alt="BlockBlast Gameplay" width="600"/>
</p>

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)

  - [Human Play](#human-play)
  - [Training Agents](#training-agents)
  - [Comparing Agents](#comparing-agents)

- [Reinforcement Learning Environment](#-reinforcement-learning-environment)
- [Algorithms](#-algorithms)
- [Results & Analysis](#-results--analysis)
- [Future Work](#-future-work)
- [Credits](#-credits)
- [License](#-license)

---

## 🚀 Features

- **Custom Gym Env**: Fine-tuned observation, action space, and rewards.
- **Action Masking**: Prevents invalid moves for PPO & DQN.
- **Multiple Agents**:
  - Random (baseline)
  - PPO (with/without masking)
  - DQN
- **Visualization**: TensorBoard logs, matplotlib plots of performance.
- **Human Interface**: Pygame-based play via keyboard & mouse.

---

## 📂 Project Structure

```
BlockBlast-Game-AI-Agent/
├── agent_comparison/      # Simulation & comparison scripts
│   ├── simulate_playing.py
│   └── results/           # CSVs & plots
├── agents/                # RL agents & models
│   ├── models/            # Saved checkpoints
│   ├── ppo_agent.py
│   └── dqn_agent.py
├── blockblast_game/       # Game environment & assets
│   ├── game_env.py
│   ├── game_renderer.py
│   ├── game_state.py
│   └── Assets/            # Sprites & images
├── human_play/            # Human‐play wrapper
│   └── human_play.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

- Python 3.8+
- [Pygame](https://www.pygame.org/news)
- [Gym](https://github.com/openai/gym)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- `pip install -r requirements.txt`

---

## 🔧 Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/RisticDjordje/BlockBlast-Game-AI-Agent.git
   cd BlockBlast-Game-AI-Agent
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scriptsctivate       # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

**Ensure you are running all commands from the root directory of the project to avoid import errors due to sibling folder structure.**

### Human Play

```bash
python -m human_play.human_play
```

- **E, R, T**: Select piece
- **Mouse hover**: Preview placement
- **Left-click / SPACE**: Place piece
- **ESC**: Restart

---

### Training Agents

#### PPO Agent

Edit `agents/ppo_agent.py` flags:

```python
train_ppo_without_masking = False
train_ppo_with_masking    = False
visualize_ppo_without_masking = False
visualize_ppo_with_masking    = False
continue_training= False
```

Run:

```bash
python -m agents.ppo_agent
```

#### DQN Agent

Edit `agents/dqn_agent.py` flags:

```python
train_dqn_agent    = False
visualize_dqn_agent = False
continue_training   = False 
```

Run:

```bash
python -m agents.dqn_agent
```

---

### Comparing Agents

1. **Simulate games**

   ```bash
   python -m agent_comparison.simulate_playing
   ```

   → CSV in `agent_comparison/results/`

2. **Visualize results**
   ```bash
   python -m agent_comparison.visualize_results
   ```
   → Performance plots in `agent_comparison/results/`

---

## 🏗 Reinforcement Learning Environment

- **Observation**:
  - 8×8 grid (0/1)
  - 3× 5×5 piece matrices
  - Current score & combo count
- **Action Space**: 3 shapes × 64 positions → 192 discrete actions
- **Reward Design** (example although I have tried many different combinations):
  - Valid placement: +0.2
  - Invalid: –2.5 (escalating penalties)
  - Line clear: +1.5 / line
  - Game over: –20

---

## 🤖 Algorithms

1. **Random Agent**
2. **PPO** (Clipped Policy Gradient)
3. **DQN** (Value-based with Replay & Target Network)
4. **Masked PPO** (invalid‐action masking)

See inline docstrings for hyperparameters and architecture details.

---

## 📊 Results & Analysis

- **DQN and PPO**: both struggle with the large action space of the game and I have not been able to get the agents to learn not to place pieces in invalid positions.
- I have solved this by introducing action masking. Masked PPO avoids invalid moves and gains higher average rewards compared to an agent that plays randomly among the valid moves.

---

## 🔭 Future Work

- Action‐masking for DQN
- Hyperparameter sweeps & longer training
- MLOps pipeline: Web UI for drawing board state & serving model (e.g., Flask + React)
- Deploy a public “BlockBlast Solver” website

---

## 🙏 Credits

- **Game Assets**: [Kefrov’s Blast](https://github.com/Kefrov/Blast/)  
  Everything else, including game logic, agent training, and analysis, is original work.

---

## 📄 License

This project is licensed under the MIT License.
