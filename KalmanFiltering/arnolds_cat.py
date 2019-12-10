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
H = np.eye(2)

# Process noise
q_var = 1e-9
Q = np.eye(2) * q_var

# Initial state
x_0 = np.array([-1.62, 1.0])
P_0 = np.eye(2) * 1e-9

def greedy_policy(s, done, w, X, env):
    q = [np.dot(w, X(s,done,a)) for a in range(env.actions)]
    return np.argmax(q)

def _eval(env, w, X):
    s, done = env.reset(), False

    g = 0.
    while not done:
        a = greedy_policy(s, done, w, X, env)
        s,r,done,_ = env.step(a)
        g += r
    return g


# Measurement noise
r_vars = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
averaged_returns_no_kf = []

for r_var in r_vars:
    R = np.eye(2) * r_var

    env = KalmanFilterWrapper(A, B, H, Q, R, x_0, P_0, False)

    X = SA(np.array([-2., -2.]),
            np.array([2., 2.]),
            3,
            num_tilings=10,
            tile_width=np.array([0.1, 0.1]))

    w = SarsaLambda(env, gamma=1., lam=0.8, alpha=0.01, X=X, num_episode=2000)
    print(f"Completed Sarsa Lambda training for noise level: {r_var}")

    trials = 100
    gs = [_eval(env, w, X) for _ in  range(trials)]
    averaged_returns_no_kf.append(sum(gs)/trials)

averaged_returns = []

for r_var in r_vars:
    R = np.eye(2) * r_var

    env = KalmanFilterWrapper(A, B, H, Q, R, x_0, P_0, True)

    X = SA(np.array([-2., -2.]),
            np.array([2., 2.]),
            3,
            num_tilings=10,
            tile_width=np.array([0.1, 0.1]))

    w = SarsaLambda(env, gamma=1., lam=0.8, alpha=0.01, X=X, num_episode=2000)
    print(f"Completed Sarsa Lambda training for noise level: {r_var}")

    trials = 100
    gs = [_eval(env, w, X) for _ in  range(trials)]
    averaged_returns.append(sum(gs)/trials)

print(averaged_returns)
print(averaged_returns_no_kf)
plt.loglog(r_vars, averaged_returns_no_kf)
plt.loglog(r_vars, averaged_returns)
plt.xlabel("Measurement noise", fontsize=14)
plt.ylabel("Average return", fontsize=14)
plt.legend(["No KF", "KF"])
plt.show()




'''
### Plot opt
x_0 = np.array([-1., 0.618 + 0.2])
R = np.eye(2) * 1e-5
env = KalmanFilterWrapper(A, B, H, Q, R, x_0, P_0, True)
while True: 
    S_p, R, done, info = env.step(A) 
    if done:
        env.reset()
        break

R = np.eye(2) * 1e-5
x_0 = np.array([0.6, -0.37082 + 0.2])
env = KalmanFilterWrapper(A, B, H, Q, R, x_0, P_0, True)
while True: 
    S_p, R, done, info = env.step(A) 
    if done:
        env.reset()
        break

R = np.eye(2) * 1e-5
x_0 = np.array([0.9, -0.5562 - 0.2])
env = KalmanFilterWrapper(A, B, H, Q, R, x_0, P_0, True)
while True: 
    S_p, R, done, info = env.step(A) 
    if done:
        env.reset()
        break

R = np.eye(2) * 1e-5
x_0 = np.array([-1.073, 0.35])
env = KalmanFilterWrapper(A, B, H, Q, R, x_0, P_0, True)
while True: 
    S_p, R, done, info = env.step(A) 
    if done:
        env.reset()
        break

R = np.eye(2) * 1e-5
x_0 = np.array([1.42, -0.5])
env = KalmanFilterWrapper(A, B, H, Q, R, x_0, P_0, True)
while True: 
    S_p, R, done, info = env.step(A) 
    if done:
        env.reset()
        break
m = -0.618034
bound = 0.5
x = np.linspace(-2., 2., 100)
ub = m * x + bound
lb = m * x - bound
plt.plot(x, m*x, 'k--')
plt.plot(x, ub, 'r--')
plt.plot(x, lb, 'r--')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
#  d_reward_circ = plt.Circle((0, 0), 0.2, color='g', alpha=0.2)
#  ax = plt.gcf().gca()
#  ax.add_artist(d_reward_circ)
plt.show()
plt.cla()
plt.clf()
'''
