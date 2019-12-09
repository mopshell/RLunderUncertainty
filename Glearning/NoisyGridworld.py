import numpy as np

from env import EnvSpec, Env, EnvWithModel
from policy import Policy

class NoisyGridWorld(EnvWithModel):  
    # GridWorld for example 4.1
    def __init__(self,noise_size):
        # 16 states: 0 and 15 terminal
        # 4 action: 0 left, 1 up, 2 right, 3 down
        env_spec = EnvSpec(16, 4, 1.)
        super().__init__(env_spec)
        self.trans_mat, self.ret_mat = self._build_trans_mat()
        self.terminal_state = [0, 15]
        self.noise_size=noise_size

    def _build_trans_mat(self):
        trans_mat = np.zeros((16, 4, 16), dtype=int)
        ret_mat = np.zeros((16, 4, 16)) - 1.

        for s in range(1, 15):
            if s % 4 == 0:
                trans_mat[s][0][s] = 1.
            else:
                trans_mat[s][0][s-1] = 1.
            if s < 4:
                trans_mat[s][1][s] = 1.
            else:
                trans_mat[s][1][s-4] = 1.
            if (s+1) % 4 == 0:
                trans_mat[s][2][s] = 1.
            else:
                trans_mat[s][2][s+1] = 1.
            if s > 11:
                trans_mat[s][3][s] = 1.
            else:
                trans_mat[s][3][s+4] = 1.

        for a in range(4):
            trans_mat[0][a][0] = 1.
            trans_mat[15][a][15] = 1.
            ret_mat[0][a][0] = 0
            ret_mat[15][a][15] = 0

        return trans_mat, ret_mat

    @property
    def TD(self):
        return self.trans_mat

    @property
    def R(self):
        return self.ret_mat+np.random.uniform(-self.noise_size,self.noise_size)

    def reset(self):
    # Random initialze location for each episode run
        self.state = np.random.randint(1, 15)
        return self.state

    def step(self, action):
        assert action in range(self.spec.nA), "Invalid Action"
        assert self.state not in self.terminal_state, "Episode has ended!"

        prev_state = self.state
        self.state = np.random.choice(self.spec.nS, p=self.trans_mat[self.state, action])
        r = self.ret_mat[prev_state, action, self.state]+np.random.uniform(-self.noise_size,self.noise_size)

        if self.state in self.terminal_state:
            return self.state, r, True
        else:
            return self.state, r, False
