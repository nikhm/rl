import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import tensorflow as tf
import numpy as np
import time
from progPrint import pprint

env = gym.make('CartPole-v0')
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
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
# Start the Training
num_episodes = 10000
num_steps = 500

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    x = np.zeros([1,num_observations])
    done_action = np.zeros([1,num_actions])
    for i_episode in range(num_episodes):
        s = env.reset()
        states, actions = [], []
        rAll = 0.0
        for t in range(num_steps):
            s = np.array([s])
            prob_a = sess.run([prob],feed_dict={input_vector:s})
            prob_a = prob_a[0]
            a = np.argmax(prob_a)
            prob_a = np.max(prob_a)
            if i_episode > num_episodes/2:
                prob_a = 0.95
            if np.random.random() > prob_a:
                a = env.action_space.sample()
            s_prime,reward,done,_ = env.step(a)
            done_action = 0*done_action
            done_action[0,a] = 1
            states.append(s)
            actions.append(done_action)
            rAll += 1
            s = s_prime
            if done:
                break
        l = len(states)
        assert l == len(actions)
        total_loss = 0.0
        # gamma = 0.99
        y = 0.99
        y = pow(y,l)
        for t in range(l):
            _,lossValue = sess.run([train,loss],feed_dict={input_vector:states[t],award:rAll,action_taken:actions[t]})
            total_loss += lossValue
            y = y/0.99
        pprint(str(total_loss))
        if (i_episode % 100 == 0):
            print '\n'
            print rAll
    save_path = saver.save(sess, "models/cartpole2.ckpt")
