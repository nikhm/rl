import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import tensorflow as tf
import numpy as np
import time
from progPrint import pprint

'''
Using Neural Network as a function approximator in Q-learning
'''


env = gym.make('FrozenLake-v0')

# Although we are still using the states, generally we use some features of the environment
# to characterize a state. Hence the name. (A vector to represent a state)
num_features = env.observation_space.n
num_actions = env.action_space.n


'''Build a model using tensorflow'''
input_vector = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))

# Q(s,a) for taking different actions
Q_a = tf.matmul(input_vector,W)

# But from Bellman equation Q(s,a) ~ r + y* max_a' Q(s',a') = Q_target
# We aim to minimize that loss
Q_target = tf.placeholder(dtype=tf.float32,shape=[1,1])
action_taken = tf.placeholder(dtype=tf.int32,shape=[1,1])
loss = tf.square(Q_a[0,action_taken[0,0]]-Q_target[0,0])
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

''' Training '''
num_episodes = 2000
max_steps = 100
lossList = []
rList = []
e = 0.1
y = 0.99

with tf.Session() as sess:
    sess.run(init)
    for i_episode in range(num_episodes):
        print 'Episode %d:\n'%(i_episode+1)
        rAll, loss_episode = 0.0, 0.0
        s = env.reset()
        done = False
        e = 1./((i_episode/50.0)+10)
        for t in range(max_steps):
            x = np.zeros([1,num_features])
            x[0,s] = 1
            Q_a_eval = sess.run([Q_a],feed_dict={input_vector:x})
            #print len(Q_a_eval)
            Q_a_eval = Q_a_eval[0]
            #print Q_a_eval
            #print Q_a_eval
            a = np.argmax(Q_a_eval[0,:])
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            #print a
            s_prime,r,done,_ = env.step(a)
            #print s_prime
            x_prime = np.zeros([1,num_features])
            x_prime[0,s_prime] = 1
            Q_new_eval = sess.run([Q_a],feed_dict={input_vector:x_prime})
            Q_new_eval = Q_new_eval[0]
            Qmax = np.max(Q_new_eval[0,:])
            # Q = r + y*Qmax
            Qt = r + y*Qmax
            Qp = Q_a_eval[0,a] # is not used however
            Qt_array = np.array([[Qt]])
            a_array = np.array([[a]])
            [_,lossValue] = sess.run([train,loss],feed_dict={input_vector:x,Q_target:Qt_array,action_taken:a_array})
            rAll += r
            print 's:%d sp:%d'%(s,s_prime)
            s = s_prime
            loss_episode += lossValue
            if done:
                break
        lossList.append(loss_episode)
        rList.append(rAll)
