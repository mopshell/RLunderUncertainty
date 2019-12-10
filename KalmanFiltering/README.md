The code for Kalman filtering based Reinforcement Learning is organized as follows:

- `true_online_sarsa.py` contains the implementation of true online Sarsa(lambda) to perform policy control.
- `state_approximators.py` contains the implementation of tile coding based feature functions for continuous state spaces
- `kalman_filter.py` handles the evolution of the dynamical system, producing observations, time stepping state estimates, and covariances.
- `arnolds_cat.py` defines the test comparing Kalman filtering vs no-filtering approaches in learning a control function for a dynamical system defined by the Arnold's cat map. The control problem is solved by true online sarsa(lambda).
