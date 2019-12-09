import numpy as np
from true_online_sarsa import SarsaLambda
from state_approximators import StateActionFeatureVectorWithTile as SA
from kalman_filter import KalmanFilterWrapper
import matplotlib.pyplot as plt

# State evolution map
J = np.array([[0, 1], [1, 1]])
dt = 2e-2
A = np.eye(2) + dt * J

# Control map
B = 0.02 * np.eye(2)

# Observation map
H = np.ones((2,2))

# Process noise
q_var = 1e-7
Q = np.eye(2) * q_var

# Measurement noise
r_var = 1e-5
R = np.eye(2) * r_var

# Initial state
x_0 = np.array([-1.62, 1.0])
P_0 = np.eye(2) * 1e-9

env = KalmanFilterWrapper(A, B, H, Q, R, x_0, P_0)

X = SA(np.array([-2., -2.]),
        np.array([2., 2.]),
        3,
        num_tilings=10,
        tile_width=np.array([0.1, 0.1]))

w = SarsaLambda(env, gamma=1., lam=0.8, alpha=0.01, X=X, num_episode=1000)

m = -0.618034
bound = 0.5
x = np.linspace(-2., 2., 100)
ub = m * x + bound
lb = m * x - bound
plt.plot(x, m*x, 'g--')
plt.plot(x, ub, 'r--')
plt.plot(x, lb, 'r--')
d_reward_circ = plt.Circle((0, 0), 0.2, color='g', alpha=0.2)
ax = plt.gcf().gca()
ax.add_artist(d_reward_circ)
plt.show()
plt.cla()
plt.clf()

def greedy_policy(s,done):
    q = [np.dot(w, X(s,done,a)) for a in range(env.actions)]
    return np.argmax(q)

def _eval(render=False):
    s, done = env.reset(), False
    if render: env.render()

    g = 0.
    while not done:
        a = greedy_policy(s,done)
        s,r,done,_ = env.step(a)
        if render: env.render()

        g += r
    return g

gs = [_eval() for _ in  range(100)]
_eval(False)

plt.plot(gs)
plt.show()
