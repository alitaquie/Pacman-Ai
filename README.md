# 🎮 AI Pacman Model

![Python](https://img.shields.io/badge/Python-3.5+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![CI](https://img.shields.io/badge/CI-passing-brightgreen)

An educational AI project implementing **Value Iteration** and **Q-Learning** agents, adapted from Berkeley’s AI Lab curriculum. You’ll first test your agents on **Gridworld**, then apply them to a **Crawler** robot controller, and finally to the full **Pac-Man** game.


## 🚀 Features

- **Value Iteration Agent**  
  Implements batch value iteration over a model MDP.  
  - Configurable iterations, discount factor, and noise  
  - Methods: `getValue`, `getQValue`, `getPolicy`  

- **Q-Learning Agent**  
  Learns entirely from interaction via the Q-learning update rule.  
  - Epsilon-greedy exploration  
  - Configurable learning rate α and exploration rate ε  

- **Approximate Q-Learning Agent**  
  Generalizes across states via feature extractors.  
  - Identity vs. custom feature extractors (e.g. SimpleExtractor)  
  - Weight updates for feature-based Q-values  

- **Environments**  
  - **Gridworld**: Classic classroom MDP with stochastic transitions  
  - **Crawler**: Simulated robot following a DAG of “rooms”  
  - **Pac-Man**: Full GUI/GIF playback for training and evaluation  

## 📦 Installation

1. Clone the repository:
  ```bash
   git clone https://github.com/ryanhui30/ai-pacman.git
   cd ai-pacman
  ```

2. Install dependencies:
 ```bash
 pip install -r requirements.txt
 ```

3. Install Tkinter for GUI:
- Linux: sudo apt-get install python3-tk
- Mac: brew install python-tk
- Windows: Comes pre-installed with Python

🕹️ Usage
Run the main game:

 ```bash
 python -m pacai.bin.pacman
 ```

Tabular Q-Learner:
 ```bash
 python3 -m pacai.bin.pacman \
  -p PacmanQAgent \
  --num-training 2000 \
  --num-games 10
 ```
Approximate Q-Learner:
 ```bash
python3 -m pacai.bin.pacman \
  -p ApproximateQAgent \
  --agent-args extractor=pacai.core.featureExtractors.SimpleExtractor \
  --num-training 50 \
  --num-games 60 \
  --layout mediumGrid
 ```
Generate Gameplay Gif:
 ```bash
 python3 -m pacai.bin.replay --output gameplay.gif
 ```

🧠 **AI Components**
- **Pathfinding:** A★ and Dijkstra algorithms for optimal route planning  
- **Strategic Decision-Making:** Minimax search with alpha-beta pruning  
- **Ghost Behavior:** Pluggable chasing, ambush, and evasion strategies  
- **Utility Evaluation:** State‐scoring and heuristic evaluation functions  
- **Model-Based Planning:** Value Iteration over the MDP  
- **Model-Free Learning:** Q-Learning with ε-greedy exploration (configurable α, γ, ε)  
- **Approximate Q-Learning:** Feature-based weight updates for generalization across states  


Need Help?
📩 Contact: alitaquiedev0@gmail.com
