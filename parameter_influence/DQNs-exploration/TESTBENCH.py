
import gym
import tensorflow as tf
import numpy as np
import random as ran
from gym import wrappers

game_name = 'Acrobot-v1'
env = gym.make(game_name).env

INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

REPLAY_QUEUE = []

HIDDEN_SIZES = [[30], [30, 30], [30, 30, 30]]  #baseline 30-30-30
LEARNING_RATES = [0.01, 0.005, 0.001, 0.0005, 0.0001] #baseline 0.001
REPLAY_LIMITS = [50000] #baseline 50k
EPOCHS = [5, 10, 20, 50] #baseline 10
BATCH_SIZES = [10 ,20, 50, 100, 200] #baseline 100
LEARNING_STEPS = [50, 100, 200, 500] #baseline 200
EXPLORATIONS = [10, 20, 30, 50] #baseline 30

MAX_EPISODE = 500
STEPS_PER_EPISODE = 5000 #5000 for acrobot
DIS = 0.99
TESTCOUNT = 50

DROP_OUT = 0.9  # not used

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
        # env.render()
        action = np.argmax(mainDQN.predict(state))
        ## stop endless episodes

        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if step > STEPS_PER_EPISODE*1: done=True; #action = 0#;print('limit')
        step += 1

        if done:
            #save test score here
            print("Total score: {}".format(reward_sum))
            break
    return reward_sum,step


def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

class DQN:
    def __init__(self, session, params, name="main"):
        self.session = session
        self._build_network(name)
        self.params = params

    def _build_network(self, name):
        with tf.variable_scope(name):
            LAYER_SIZE = [INPUT_SIZE] + params['hidden_size'] + [OUTPUT_SIZE]

            W_list = []
            for i in range(len(LAYER_SIZE)-1):
                W = tf.get_variable(str.format("W{}", i),
                                    shape=[LAYER_SIZE[i], LAYER_SIZE[i+1]],
                                    initializer=tf.contrib.layers.xavier_initializer())
                W_list.append(W)

            self.x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
            layer = self.x
            for W in W_list[:-1]:
                layer = tf.nn.tanh(tf.matmul(layer, W))

            self.y_pred = tf.matmul(layer, W_list[-1])

            print('{} Hidden Layer'.format(len(params['hidden_size'])))

        self.dropout = tf.placeholder(dtype=tf.float32) # not used

        self.y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

        self.loss = tf.reduce_sum(tf.square(self.y_pred - self.y))
        self.train = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(self.loss)

    def predict(self, input_data):
        input_data = np.reshape(input_data, (1, INPUT_SIZE))
        return self.session.run(self.y_pred, feed_dict={self.x: input_data, self.dropout: DROP_OUT})

    def update(self, input_data, output_data):
        return self.session.run([self.loss, self.train], feed_dict={self.x: input_data, self.y:output_data, self.dropout: DROP_OUT})


params = {
    'learning_rate': LEARNING_RATES[2],
    'replay_limit': REPLAY_LIMITS[0],
    'exploration': EXPLORATIONS[0],
    'epoch': EPOCHS[1],
    'mini_batch_size': BATCH_SIZES[3],
    'learning_step': LEARNING_STEPS[2],
    'hidden_size': HIDDEN_SIZES[2]
}

primer = 0
for h in EXPLORATIONS:  ## change here
    primer += 1 
    params['exploration'] = h ## change here
    for i in range(10):
        tf.reset_default_graph()

        with tf.Session() as sess:

            mainDQN = DQN(sess, params=params, name='main')
            targetDQN = DQN(sess, params=params, name="target")

            ############### TEST LOGS ##############

            ###### train logs #####
            train_ERL = []  # episode reward log, logs reward after every episode
            train_ESL = []  # episode step log, logs how many steps before done
            train_L =  []   # training loss after each update

            ###### play logs #####
            play_ERL = []   # eposide reward log, logs reward after every episode
            play_ESL = []   # episode step log, logs how many steps before done

            print('RUN RUN RUN {}'.format(i))
            tf.global_variables_initializer().run()
            copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
            sess.run(copy_ops)

            for episode in range(MAX_EPISODE):
                state = env.reset()
                reward_episode = 0
                done = False
                e = 1. / ((episode/params['exploration'])+1) # Lunar 30 / Acro 50
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

                    if len(REPLAY_QUEUE) > params['replay_limit']:
                        del REPLAY_QUEUE[0]
                    state = next_state
                    if step_count > STEPS_PER_EPISODE:
                        break

                print("Episode {} steps {} with Reward {}".format(episode, step_count, reward_episode))
                train_ERL += [reward_episode]
                train_ESL += [step_count]

                if episode % params['epoch'] == params['epoch']-1:
                    for _ in range(params['learning_step']):
                        mini_batch = ran.sample(REPLAY_QUEUE, params['mini_batch_size'])
                        loss, _ = ddqn_replay_train(mainDQN, targetDQN, mini_batch)
                    print("loss : {}".format(loss))
                    train_L += [loss]
                sess.run(copy_ops)

            # See our trained bot in action
            #env2 = wrappers.Monitor(env, 'gym-results', force=True)
            for f in range(TESTCOUNT):
                a,b = bot_play(mainDQN, env=env)
                play_ESL += [b]
                play_ERL += [a]

            ############### SAVE LOGS ##############
            train_ERL = np.array(train_ERL)
            train_ESL = np.array(train_ESL)
            train_L =  np.array(train_L)

            play_ERL = np.array(play_ERL)
            play_ESL = np.array(play_ESL)

            ### saveing
            np.save('log/train/{}_train_ERL_{}'.format(primer, i),train_ERL)
            np.save('log/train/{}_train_ESL_{}'.format(primer, i),train_ESL)
            np.save('log/train/{}_train_L_{}'.format(primer, i),train_L)

            np.save('log/play/{}_play_ERL_{}'.format(primer, i),play_ERL)
            np.save('log/play/{}_play_ESL_{}'.format(primer, i),play_ESL)
        print('saving done...')
