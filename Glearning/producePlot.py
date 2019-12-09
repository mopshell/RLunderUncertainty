import qandglearning as qgl
import matplotlib.pyplot as plt
from NoisyGridworld import NoisyGridWorld
import numpy as np
from policy import Policy

class RandomPolicy(Policy):
    def __init__(self, nA, p=None):
        self.p = p if p is not None else np.array([1/nA]*nA)

    def action_prob(self, state, action=None):
        return self.p[action]

    def action(self, state):
        return np.random.choice(len(self.p), p=self.p)

def producePlot(noise_size):
    env=NoisyGridWorld(noise_size)
    done=False
    state=env.reset()
    rho=RandomPolicy(4)
    Q,pi,Qrewards=qgl.Qlearning(env,0.1,0.1,np.zeros((16,4)),1000)
    Grewards=[]
    G=np.zeros((16,4))
    for i in range(1000):
        G,pi,reward=qgl.Glearning(env,rho,0.1,G,(i+1)/100,1)
        Grewards.append(reward[0])
    Qrewards,Grewards=np.array(Qrewards),np.array(Grewards)
    plt.plot(Qrewards-Grewards)
    plt.show()
    
def runExperiment(num_ensembles,noise_step_size,num_noise_steps,run_length):
    cum_diffs=[]
    for noise_size in range(num_noise_steps):
        noise_size=noise_size*noise_step_size
        env=NoisyGridWorld(noise_size)
        done=False
        state=env.reset()
        rho=RandomPolicy(4)
        totQreward=0
        for iterations in range(num_ensembles):
            Q,pi,Qrewards=qgl.Qlearning(env,0.1,0.1,np.zeros((16,4)),run_length)
            totQreward+=sum(Qrewards)
        totQreward=totQreward/num_ensembles
        totGreward=0
        G=np.zeros((16,4))
        for iterations in range(num_ensembles):
            for i in range(run_length):
                G,pi,reward=qgl.Glearning(env,rho,0.1,G,(i+1)/10,1)
                totGreward+=reward[0]
        totGreward=totGreward/num_ensembles
        cum_diffs.append(totQreward-totGreward)
    plt.plot(np.linspace(0,(num_noise_steps-1)*noise_step_size,num_noise_steps),cum_diffs)
    plt.show()
