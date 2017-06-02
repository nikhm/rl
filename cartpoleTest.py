import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import tensorflow as tf
import numpy as np
import time
from progPrint import pprint
from gym import wrappers

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env,'tmp/cartPole-v0-2')
s = env.reset()

num_observations = len( env.reset() );
num_actions = env.action_space.n;


# Build the model
hidden_layer_size = 10
input_vector = tf.placeholder(shape=[1,num_observations],dtype=tf.float32)
W1 = tf.get_variable( "W1", shape = [num_observations,num_actions] , initializer = tf.contrib.layers.xavier_initializer() )
h = tf.nn.relu(tf.matmul(input_vector,W1))
prob = tf.nn.softmax(h)

# Required here
tv = tf.trainable_variables()
regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv if 'bias' not in v.name])*0.005

action_taken = tf.placeholder( shape=[1,num_actions], dtype=tf.float32 )
award = tf.placeholder(shape=[],dtype=tf.float32)
loss = -tf.reduce_sum(tf.log( 1e-10 + tf.multiply(action_taken,prob) )*award) # + regularization_cost
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss)

sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, "models/cartpole2.ckpt")

init = tf.global_variables_initializer()

num_steps = 500
rAll = 0

num_episodes = 200

for i_episode in range(num_episodes):
    s = env.reset()
    for t in range(500):
        s = [s]
        env.render()
        #time.sleep(0.01)
        prob_a = sess.run([prob],feed_dict={input_vector:s})
        prob_a = prob_a[0]
        a = np.argmax(prob_a)
        s_prime,reward,done,_ = env.step(a)
        rAll += reward
        s = s_prime
        if done:
            break


print rAll
