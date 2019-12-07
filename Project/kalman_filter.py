class KalmanFilterWrapper:
    def __init__(self, noisy_env, other_params):
        self.env = noisy_env
        self.S = None

    def aposteriori(measurement):
        return self.S

    def reset():
        S = self.env.reset()
        # TODO: reset filter?
        return self.aposteriori(S)

    def step(self):
        (S, R, done, info) = self.env.step()
        return (self.aposteriori(S), R, done, info)
