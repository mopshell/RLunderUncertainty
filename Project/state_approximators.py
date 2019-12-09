import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_dim = state_low.shape[0]
        self.X_dim = np.zeros((self.state_dim+2,), dtype=int)
        self.X_dim[0] = num_actions
        self.X_dim[1] = num_tilings
        self.X_dim[2:] = np.ceil(np.divide((state_high - state_low), tile_width)) + 1
        #  self.tiles_per_dim = np.ceil(np.divide((state_high - state_low), tile_width)) + 1
        self.tw = tile_width
        self.num_tilings = num_tilings
        self.num_actions = num_actions
        tiling_index = np.zeros((self.state_dim, num_tilings))
        displacement = np.array([i for i in range(2*self.state_dim) if i%2==1])
        #  displacement = np.ones((self.state_dim,))

        for i in range(self.num_tilings):
            tiling_index[:, i] = i * displacement

        fu = tile_width/num_tilings
        fu = tile_width / (tiling_index[-1, -1]+1)

        self.offsets = state_low.reshape((self.state_dim, 1)) \
                        - tiling_index * fu.reshape((self.state_dim, 1))

        self.dim = np.product(self.X_dim)

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.dim

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        X = np.zeros(self.X_dim)
        for n in range(self.num_tilings):
            X_ = X[a][n]
            for d in range(self.state_dim):
                begin = self.offsets[d, n]
                idx = int((s[d] - begin)/self.tw[d])
                if (d+1) == self.state_dim:
                    X_[idx] = 1
                else:
                    X_ = X_[idx]

        return X.reshape((self.dim,))
