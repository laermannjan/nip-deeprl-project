
import gym
import tensorflow as tf
import numpy as np
import random as ran
from gym import wrappers


game_name = 'Acrobot-v1'
env = gym.make(game_name).env

INPUT_SIZE = env.observation_space.shape[0]
LAYER = 3 ; layerrange = [1,2,3]
HIDDEN_SIZE = [35, 30, 25]
OUTPUT_SIZE = env.action_space.n
LEARNING_RATE = 0.001 ; lraterange = [0.01,0.005,0.001,0.0005,0.0001]

REPLAY_QUEUE = []
REPLAY_LIMIT = 50000 ; replayrange = [1,5,10,20]

MAX_EPISODE = 500 #maybe 5k?10k? 
STEPS_PER_EPISODE = 500
EPISODE_PER_TRAINING = 10 ; epochrange = [5,10,20,50]
MINI_BATCH_SIZE = 100 ; batchrange = [20,50,100,200]
LEARNING_STEPS = 200 ; steprange = [50,100,200,500]

DROP_OUT = 0.9  # not used
DIS = 0.99
EXPLORATION = 50 ; epsilonrange = [30,50,100,200]

TESTCOUNT = 100

def ddqn_replay_train(mainDQN, targetDQN, train_batch):
    '''
    Double DQN implementation
    :param mainDQN: main DQN
    :param targetDQN: target DQN
    :param train_batch: minibatch for train
    :return: loss
    '''
    x_stack = np.empty(0).reshape(0, INPUT_SIZE)
    y_stack = np.empty(0).reshape(0, OUTPUT_SIZE)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        # print(state, action, reward)
        Q = mainDQN.predict(state)
        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # Double DQN: y = r + gamma * targetDQN(s')[a] where
            # a = argmax(mainDQN(s'))
            Q[0, action] = reward + DIS * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)
def bot_play(mainDQN, env=env):
    # See our trained network in action
    state = env.reset()
    reward_sum = 0
    step = 0
    while True:
        #env.render()
        action = np.argmax(mainDQN.predict(state))
        ## stop endless episodes
        if step > STEPS_PER_EPISODE*10: action = 0#;print('limit')
        step += 1

        state, reward, done, _ = env.step(action)
        reward_sum += reward
        
        if done:
            #save test score here
            print("Total score: {}".format(reward_sum))
            break
def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):

    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder
class DQN:
    def __init__(self, session, name="main"):
        self.session = session
        self._build_network(name)


    def _build_network(self, name):
        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, shape = [None, INPUT_SIZE])
            W1 =  tf.get_variable("W1", shape = [INPUT_SIZE, HIDDEN_SIZE[0]], initializer=tf.contrib.layers.xavier_initializer())
            W2 = tf.get_variable("W2", shape = [HIDDEN_SIZE[0], HIDDEN_SIZE[1]], initializer=tf.contrib.layers.xavier_initializer())
            W3 = tf.get_variable("W3", shape = [HIDDEN_SIZE[1], HIDDEN_SIZE[2]], initializer=tf.contrib.layers.xavier_initializer())
            W4 = tf.get_variable("W4", shape = [HIDDEN_SIZE[2], OUTPUT_SIZE], initializer=tf.contrib.layers.xavier_initializer())

            layer1 = tf.nn.tanh(tf.matmul(self.x, W1))
            layer2 = tf.nn.tanh(tf.matmul(layer1, W2))
            layer3 = tf.nn.tanh(tf.matmul(layer2, W3))

            if LAYER == 1: 
                W2_ = tf.get_variable("W2_", shape = [HIDDEN_SIZE[0], OUTPUT_SIZE], initializer=tf.contrib.layers.xavier_initializer())
                self.y_pred = tf.matmul(layer1, W2_)
                print('1 hidden layer')
            elif LAYER == 2: 
                W3_ = tf.get_variable("W3_", shape = [HIDDEN_SIZE[1], OUTPUT_SIZE], initializer=tf.contrib.layers.xavier_initializer())
                self.y_pred = tf.matmul(layer2, W3_)
                print('2 hidden layer')
            else: 
                self.y_pred = tf.matmul(layer3, W4)
                print('3 hidden layer')

        self.dropout = tf.placeholder(dtype=tf.float32) # not used
                
        self.y = tf.placeholder(tf.float32, shape = [None, OUTPUT_SIZE])

        self.loss = tf.reduce_sum(tf.square(self.y_pred - self.y))
        self.train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.loss)

    def predict(self, input_data):
        input_data = np.reshape(input_data, (1, INPUT_SIZE))
        return self.session.run(self.y_pred, feed_dict={self.x: input_data, self.dropout: DROP_OUT})

    def update(self, input_data, output_data):
        return self.session.run([self.loss, self.train], feed_dict={self.x: input_data, self.y:output_data, self.dropout: DROP_OUT})

with tf.Session() as sess:

    mainDQN = DQN(sess)
    targetDQN = DQN(sess, name="target")
    tf.global_variables_initializer().run()
    copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
    sess.run(copy_ops)

    for episode in range(MAX_EPISODE):
        state = env.reset()
        reward_episode = 0
        done = False

        e = 1. / ((episode/EXPLORATION)+1) ##Lunar 30 / Acro 50
        step_count = 0
        while not done:
            step_count += 1
            if np.random.rand(1) < max(0.1,e):
                action = env.action_space.sample()
            else:
                action = np.argmax(mainDQN.predict(state))

            next_state, reward, done, _ = env.step(action)

            if done:
                reward = 30
                    
            reward_episode += reward
            REPLAY_QUEUE.append([state, action, reward, next_state, done])

            if len(REPLAY_QUEUE) > REPLAY_LIMIT:
                del REPLAY_QUEUE[0]

            state = next_state
            if step_count > STEPS_PER_EPISODE:
                break

        print("Episode {} steps {} with Reward {}".format(episode, step_count, reward_episode))

        if episode % EPISODE_PER_TRAINING == EPISODE_PER_TRAINING-1:
            
            for _ in range(LEARNING_STEPS):
                mini_batch = ran.sample(REPLAY_QUEUE, MINI_BATCH_SIZE)
                loss, _ = ddqn_replay_train(mainDQN, targetDQN, mini_batch)

            print("loss : {}".format(loss))
            sess.run(copy_ops)

    # See our trained bot in action
    #env2 = wrappers.Monitor(env, 'gym-results', force=True)

    for i in range(TESTCOUNT):
        bot_play(mainDQN, env=env)







