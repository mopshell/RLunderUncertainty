import numpy as np
import matplotlib.pyplot as plt

class KalmanFilterWrapper:
    def __init__(self, A, B, H, Q, R, x_0, P_0, estimation=True):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x_hat = x_0 + np.random.multivariate_normal(np.zeros(x_0.shape), P_0)
        self.x_k_m = x_0
        self.z_k = np.dot(H, x_0)
        self.P = P_0
        self.s_dim = x_0.shape[0]
        self.x_0 = x_0
        self.P_0 = P_0
        self.n_time_steps = 0
        self.actions = 3
        self.ev = [x_0]
        self.estimation = estimation

    def reset(self):
        x_0 = self.x_0 + np.random.randn(2) * 0.05
        self.x_hat = x_0 + np.random.multivariate_normal(np.zeros(x_0.shape), self.P_0)
        self.x_k_m = x_0
        self.z_k = np.dot(self.H, x_0)
        #  self.P = self.P_0
        self.n_time_steps = 0
        self.ev = [x_0]
        return self.x_hat 

    def action_to_control(self, action):
        '''
        Action to control mapping:
            0 - move left
            1 - no control
            2 - move right
        '''
        #  print(f"Action picked: {action}")
        m = -0.618034
        mx = m * self.x_k_m[0]
        if (self.x_k_m[1] > mx):
            return np.array([-1, 0.])
        elif np.allclose(self.x_k_m[1], mx):
            return np.array([0, 0.])
        else:
            return np.array([1, 0.])
        #  return np.array([action-1, 0.])

    def within_valley(self, state):
        m = -0.618034
        bound = 0.5
        ub = m * state[0] + bound
        lb = m * state[0] - bound
        return (state[1] < ub) and (state[1] > lb)

    def step(self, action):
        self.time_update(self.action_to_control(action))
        self.n_time_steps += 1

        done = False
        reward = 1

        #  if np.linalg.norm(self.x_k_m) < 0.2:
            #  reward = 5.

        # If outside 2x(unit ball), controller failed. Reward -1
        #  if np.linalg.norm(self.x_k_m) > 2.0 or self.n_time_steps == 400:
        if not self.within_valley(self.x_k_m) or self.n_time_steps == 400:
            #  print(f"Episode terminated: x = {self.x_k_m}, t = {self.n_time_steps}")
            ev_mat = np.array(self.ev)
            plt.plot(ev_mat[:,0], ev_mat[:, 1])
            done = True

        info = None

        if self.estimation:
            return (self.x_hat, reward, done, info) 
        else:
            return (self.x_k_m, reward, done, info)

    def update_dynamical_system(self, u):
        # Draw process noise
        w_k_m = np.random.multivariate_normal(np.zeros(self.x_k_m.shape), self.Q)

        # Update dynamical system
        x_k = np.dot(self.A, self.x_k_m) + np.dot(self.B, u) + w_k_m
        self.ev.append(x_k)
        #  print(f"State_prev: {self.x_k_m}, state: {x_k}, t: {self.n_time_steps}")

        # Draw measurement noise
        v_k = np.random.multivariate_normal(np.zeros(self.z_k.shape), self.R)

        # Obtain measurement
        self.z_k = np.dot(self.H, x_k) + v_k
        self.x_k_m = x_k


    def time_update(self, u):
        '''
        Control variable u
        '''
        x_prior = np.dot(self.A, self.x_hat) + np.dot(self.B, u)
        P_prior = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        self.update_dynamical_system(u)

        # Compute Kalman gain
        S = self.H @ P_prior @ self.H.T + self.R
        K_k = np.linalg.solve(S.T, self.H @ P_prior.T).T

        # Update a posteriori estimate
        self.x_hat = K_k @ (self.z_k - self.H @ x_prior)
        self.P = (np.eye(self.s_dim) - K_k @ self.H) @ P_prior

