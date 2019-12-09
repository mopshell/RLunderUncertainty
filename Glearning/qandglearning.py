import numpy as np
from env import Env,EnvSpec,EnvWithModel
from policy import Policy

class EpsilonDeterministicPolicy(Policy):
    def __init__(self,nS,nA,epsilon,pol):
        self.nS=nS
        self.nA=nA
        self.pol=pol
        self.epsilon=epsilon
        
    def action_prob(self,state,action):
        if self.pol[state]!=action:
            return self.epsilon/self.nA
        return 1-self.epsilon+self.epsilon/nA
        
    def action(self,state):
        randomAction=np.random.random()<self.epsilon
        if randomAction:
            return np.random.randint(0,self.nA)
        return self.pol[state]

class ArbitraryPolicy(Policy):
    def __init__(self,nS,nA,pol):
        self.nS=nS
        self.nA=nA
        self.pol=pol
        
    def action_prob(self,state,action):
        return self.pol[state][action]
    
    def action(self,state):
        return np.random.choice(self.nA,p=self.pol[state])

def Qlearning(
    env:EnvWithModel,
    epsilon:float,
    alpha:float,
    initQ:np.array,
    num_episodes:int
):
    rewards=[]
    env_spec=env.spec
    nS=env_spec.nS
    nA=env_spec.nA
    gamma=env_spec.gamma
    Q=initQ.copy()
    pi=EpsilonDeterministicPolicy(nS,nA,epsilon,np.argmax(Q,1))
    for _ in range(num_episodes):
        state=env.reset()
        done=False
        t=0
        totalReward=0
        while not done:
            action=pi.action(state)
            new_state,reward,done=env.step(action)
            Q[state][action]+=alpha*(reward+gamma*np.max(Q[new_state])-Q[state][action])
            pi=EpsilonDeterministicPolicy(nS,nA,epsilon,np.argmax(Q,1))
            state=new_state
            totalReward+=reward*(gamma**t)
            t+=1
        rewards.append(totalReward)
    return Q,pi,rewards
    
def Glearning(
    env:EnvWithModel,
    rho:Policy,
    alpha:float,
    initG:np.array,
    beta:float,
    num_episodes:int
):
    env_spec=env.spec
    nS,nA,gamma=env_spec.nS,env_spec.nA,env_spec.gamma
    G=initG.copy()
    pol=np.zeros((nS,nA))
    for state in range(nS):
        elts=[0 for i in range(nA)]
        for action in range(nA):
            elts[action]=rho.action_prob(state,action)*np.exp(-beta*G[state,action])
        pol[state]=np.array(elts)/np.sum(np.array(elts))
    pi=ArbitraryPolicy(nS,nA,pol)
    rewards=[]
    for _ in range(num_episodes):
        state=env.reset()
        done=False
        totalReward=0
        t=0
        while not done:
            action=pi.action(state)
            new_state,reward,done=env.step(action)
            cost=-reward
            new_amount=0
            for i in range(nA):
                new_amount+=rho.action_prob(new_state,i)*np.exp(-beta*G[new_state][i])
            update=cost-(gamma/beta)*np.log(new_amount)
            G[state][action]=(1-alpha)*G[state][action]+alpha*update
            elts=[0 for action in range(nA)]
            for i in range(nA):
                elts[i]=rho.action_prob(state,i)*np.exp(-beta*G[state][i])
            pol[state]=np.array(elts)/np.sum(np.array(elts))
            pi=ArbitraryPolicy(nS,nA,pol)
            state=new_state
            totalReward+=reward*(gamma**t)
            t+=1
        rewards.append(totalReward)
    return G,pi,rewards
