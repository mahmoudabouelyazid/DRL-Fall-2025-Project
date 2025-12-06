# DRL-Fall-2025-Project
Decentralized Multi-Agent PPO for Dynamic Task Allocation and Coordination in Grid-Based Warehouse Environments


# Training and Evaluation Guide

## Dependencies

Install the required packages using pip:
```bash
pip install numpy torch pettingzoo gymnasium matplotlib
```

## Usage

To train the RL model, run `python custom_environment_train.py` (or `python custom_environment_train.py --train`) which will train a PPO-based actor-critic network for 500 updates using curriculum learning (starting with fewer tasks and gradually increasing difficulty), and save the trained model to `rl_model.pth` in the current directory. The training process uses hyperparameters optimized for the warehouse environment including a learning rate of 3e-4, entropy coefficient of 0.05, and GAE-lambda of 0.95, with checkpoints saved every 50 updates. Once training is complete, run `python algorithm_comparison.py` to compare the trained RL model against ACO and Greedy algorithms on the same environment configuration; the script will automatically detect `rl_model.pth` if it exists in the current directory. The comparison runs 10 episodes per algorithm by default (configurable with `--episodes N`) and reports comprehensive metrics including total deliveries, idle time percentage, total rewards, and late task metrics (late deliveries, late delivery percentage, average/max delivery priority, and total late penalty) to evaluate how well each algorithm handles the priority system where tasks become "late" when their priority reaches 100.

