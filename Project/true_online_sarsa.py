import numpy as np
from state_approximators import StateActionFeatureVectorWithTile

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.1):
        nA = env.actions
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))
    gl = gamma * lam

    for eps in range(num_episode):
        S = env.reset()
        A = epsilon_greedy_policy(S, False, w)
        x = X(S, False, A)
        z = np.zeros((X.feature_vector_len()))
        Q_old = 0

        while True: 
            S_p, R, done, info = env.step(A) 
            A_p = epsilon_greedy_policy(S_p, done, w)
            x_p = X(S_p, done, A_p)
            Q = np.dot(w, x)
            Q_p = np.dot(w, x_p)
            delta = R + gamma * Q_p - Q
            z = gl * z + (1 - alpha * gl * np.dot(z, x)) * x
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_p
            x = x_p
            A = A_p

            if done:
                break

    return w
