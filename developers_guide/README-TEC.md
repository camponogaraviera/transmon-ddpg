<div align='center'>
  <h1> Project Architecture & Technology Stack </h1>
</div>

# Table of Contents

- [File Structure Tree](#file-structure-tree)
- [Dependencies](#dependencies)

---

# File Structure Tree

```bash
artifacts/
├── model_and_weights/
│   ├── actor_ddpg.weights.h5
│   ├── critic_ddpg.weights.h5
│   ├── target_actor_ddpg.weights.h5
│   └── target_critic_ddpg.weights.h5

assets/
├── plots/
│   ├── score_trend.png
│   └── transmon_regime.png
├── anharmonicity.gif
└── coherence.gif

configs/
└── config.cfg

developers_guide/
└── README-TEC.md

reinforce_transmon/
├── scripts/
│   └── transmon_regime.py
├── src/
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── transmon.py
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── networks.py
│   │   ├── replay_buffer.py
│   │   ├── trainer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── check_env.py
│   │   ├── gif.py
│   │   ├── normalization.py
│   │   ├── plotting.py
│   ├── __init__.py
│   └── main.py
├── VERSION.txt
├── __init__.py
├── cli.py
└── sanity.py

.editorconfig
.flake8
.gitattributes
.gitignore
.pre-commit-config.yaml
LICENSE.md
README.md
cpu_environment.yml
gpu_environment.yml
pyproject.toml
setup.py
```

---

# Dependencies

- `TensorFlow 2`: deep learning framework to build actor-critic neural networks and implement the DDPG agent class.
- `gymnasium`: used to ensure compatibility with Gym-style environments.
- `scQubits`: core library for simulating superconducting qubits, including transmon properties such as anharmonicity and coherence time.
- `matplotlib`: library for plotting and visualizing data (e.g., reward evolution and inference results).
- `imageio`: library for reading and writing image and video files in various formats.
- `numpy`: core library for numerical operations and array manipulation.
- `pytest`: framework for writing and running unit tests.
- `black`: opinionated code formatter for consistent Python style.
- `flake8`: linting tool for enforcing style and catching errors.
- `pre-commit`: framework for managing Git pre-commit hooks to enforce code quality checks before commits.
