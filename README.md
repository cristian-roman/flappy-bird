# Flappy Bird RL – Q-Learning & Deep Q-Learning Implementation

This project implements an AI agent that learns to play **Flappy Bird** using both **Q-Learning** and **Deep Q-Learning (DQL)** techniques in Python.

---

## 🗂️ Project Structure

```
flappy-bird-main/
├── constants.py         # Game constants and environment setup
├── dql_model.py         # Deep Q-Learning model (likely using PyTorch)
├── q_learn.py           # Tabular Q-learning algorithm
├── learning_main.py     # Training entry point
├── eval_main.py         # Evaluation script for trained agents
├── playable_main.py     # Script to manually play the game
├── log.py               # Logging utilities
└── README.md            # Existing documentation
```

---

## 🧠 Techniques & Technologies

### ✅ Q-Learning (Tabular)
- Implemented in `q_learn.py`
- Used for simple discrete state-action mappings

### ✅ Deep Q-Learning
- `dql_model.py` implements a neural network approximator (likely PyTorch or NumPy-based)
- Allows the agent to generalize across large or continuous state spaces

### ✅ Logging & Evaluation
- `log.py` captures learning metrics
- `eval_main.py` evaluates trained policies

---

## 🛠️ Stack & Tools

- **Language**: Python 3
- **Libraries**:
  - Likely `numpy` for matrix operations
  - Possibly `torch` or `tensorflow` (check `dql_model.py`)
- Minimal dependencies – focused on logic rather than heavy frameworks

---

## ▶️ How to Use

### Train a Q-Learning Agent:
```bash
python3 learning_main.py --mode qlearn
```

### Train a Deep Q-Learning Agent:
```bash
python3 learning_main.py --mode dql
```

### Evaluate the Agent:
```bash
python3 eval_main.py
```

### Play the Game Manually:
```bash
python3 playable_main.py
```

> You may need to install additional requirements depending on your environment (e.g., `pygame` or `torch`).

---

## 🎮 Educational Focus

This project is ideal for learning:
- How reinforcement learning works in game environments
- Difference between classical Q-learning and DQL
- How to design environments for training RL agents
