import gym
from noisy_env import NoisyEnv
from kalman_filter import KalmanFilter
from true_online_sarsa import SarsaLambda
from state_approximators import StateActionFeatureVectorWithTile

env = gym.make("MountainCar-v0") 
noisy_env = NoisyEnv(env, state_noise, measurement_operator)
kf = KalmanFilterWrapper(noisy_env) 
gamma = 1.
alpha = 3e-4
num_episodes = 1000
X = StateActionFeatureVectorWithTile(
                 env.observation_space.low,
                 env.observation_space.high,
                 env.action_space.n,
                 num_tilings=10,
                 tile_width=np.array([0.45, 0.035]))

w_star = SarsaLambda(
    kf, # openai gym environment
    gamma, # discount factor
    lam, # decay rate
    alpha, # step size
    X,
    num_episodes)
