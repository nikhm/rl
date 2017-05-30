import numpy as np
import tensorflow as tf
import gym
import random
import time
from progPrint import pprint


# Q-learning with a table
# Useful for situations with less number of states x actions

env = gym.make('FrozenLake-v0')

# The reset method returns observation of environment. ex: state
s = env.reset() # observation of start state


num_actions = env.action_space.n
num_observations = env.observation_space.n

print 'num_actions: ' + str(num_actions)
print 'num_observations: ' + str(num_observations)

# Build a q table with 0s
Q = np.zeros([num_observations,num_actions])

# Algorithm for obtaining Q-values for above table
num_episodes = 8000
num_steps = 100
e = 1.0
lr = 0.9
y = 0.95

rList = []

for i_episode in range(num_episodes):
    e = e/(i_episode + 1)
    s = env.reset()
    done = False
    r_All = 0.0
    for t in range(num_steps):
        a = np.argmax( Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i_episode+1)) )
        s_prime,r,done,_ = env.step(a)
        Q[s,a] = Q[s,a] + lr*( r + y*np.max(Q[s_prime,:]) - Q[s,a])
        s = s_prime
        r_All += r
        if done:
            break
    out = "Reward for episode %d : "%(i_episode + 1) + str(r_All)
    pprint(out)
    rList.append(r_All)

print '\n'
print Q
print str(sum(rList[:num_episodes/3]))
print str(sum(rList[num_episodes/3:2*num_episodes/3]))
print str(sum(rList[2*num_episodes/3:num_episodes]))
print str(sum(rList[:num_episodes]))

# We observe the Q values seem to improve over time and then converge (observed for ~3500 iterations)

# So let's see how an episode runs
s = env.reset()
r_All = 0.0
for t in range(num_steps):
    a = np.argmax(Q[s,:])
    s,r,done,_ = env.step(a)
    env.render()
    r_All += r
    if done:
        break

if r_All >= 1.0:
    print "Won the game"
else:
    print "Lost the game"
