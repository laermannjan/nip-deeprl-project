class RandomAgent(object):
    """Agent acting randomly."""
    def __init__(self, action_space):
        self.action_space = action_space
        self.config = None

    def act(self):
        return self.action_space.sample()

    def play(self, env, sess):
        state = env.reset()
        reward_sum = 0
        while True:
            action = self.act()
            state, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                return reward_sum
