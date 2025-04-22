---

# Tetris AI (MDP & Neural Approaches)

This project contains a Tetris-playing agent implemented using both MDP (Markov Decision Process) and Q-learning with neural networks.

## 📁 Code Structure

```
├── Model Stats/         # Contains performance statistics for trained models
├── Model Training/      # Scripts for training neural models
├── Models/              # Saved models (e.g. .pt or .pkl files)
├── __pycache__/         # Python bytecode cache
├── MDP.py               # Core MDP algorithm implementation
├── MDPBlockDrop.py      # MDP simulation for falling blocks
├── TetrisEnv.py         # Tetris environment definition
├── qlnn.py              # Q-learning with neural network
├── run.py               # Main script to run experiments
├── plot.py              # Visualization utilities
├── env.yml              # Conda environment & dependencies
├── README.md            # You're here :)
```

## 📦 Setup

Install dependencies using `conda`:

```bash
conda env create -f env.yml
conda activate tetris
```

## ✅ Running

To train or evaluate models:

```bash
python run.py
```

Or experiment with MDP:

```bash
python MDP.py
```

---
