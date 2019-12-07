import gym

class NoisyEnv:
    def __init__(self, env, state_noise, m_op):
        '''
        Initialize a noisy environment with a OpenAI gym 
        environment, a state noise generator and a measurement operator

        Args:
            env - OpenAI gym env
            state_noise - additive noise function to state
            m_op - measurement operator mapping from state to noisy measurement
        '''
        self.env = env
        self.state_noise = state_noise
        self.m_op = m_op

    def reset(self):
        return self.m_op(self.env.reset() + self.state_noise())

    def step(self):
        (S, R, done, info) = self.env.step()
        S_noisy = S + self.state_noise()
        return (self.m_op(S_noisy), R, done, info)

