import tensorflow as tf, numpy as np, gym, random
import config
import matplotlib.pyplot as plt


class QNetwork():
    # This class essentially defines the network architecture.

    def __init__(self, env):
        # Main network architecture, loss calculation and optimizer
        self.tf_sess = None
        self.saver = None
        num_observations = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # Model
        if config.extractor_type == 'conv':
            self.states = tf.placeholder(name='states', shape=(None, 84, 84, 4), dtype=tf.float32)
        else:
            self.states = tf.placeholder(name='states', shape=(None, num_observations), dtype=tf.float32)

        with tf.variable_scope('main'):
            self.features, feat_length = config.extractor(self.states, num_observations, config.extractor_type)
            self.Q = config.estimate_Q(self.features, input_size=feat_length, num_actions=num_actions,
                                       dueling=config.dueling)

        # Target network for double Q learning (space invaders)
        if config.double:
            with tf.variable_scope('target'):
                self.tar_features, feat_length = config.extractor(self.states, num_observations, config.extractor_type)
                self.tar_Q = config.estimate_Q(self.tar_features, input_size=feat_length, num_actions=num_actions,
                                           dueling=config.dueling)
        # Loss
        self.Q_target = tf.placeholder(name='Q_target', shape=(None,), dtype=tf.float32)
        self.actions = tf.placeholder(name='actions', shape=(None,), dtype=tf.int32)
        actions_one_hot = tf.one_hot(self.actions, depth=num_actions)
        self.Q_pred = tf.reduce_sum(actions_one_hot * self.Q, axis=1)
        self.loss = tf.reduce_mean((self.Q_target - self.Q_pred) ** 2)

        # Optimizer
        self.optim = tf.train.AdamOptimizer(learning_rate=config.lr).minimize(self.loss)

    def update_target(self):
        # Synchronization step for double Q learning (space invaders)
        # Get trainable variables
        trainable_variables = tf.trainable_variables()
        # Main net variables
        trainable_variables_main = [var for var in trainable_variables if var.name.startswith('main')]
        # Target net variables
        trainable_variables_target = [var for var in trainable_variables if var.name.startswith('target')]
        for i in range(len(trainable_variables_main)):
            self.tf_sess.run(tf.assign(trainable_variables_target[i], trainable_variables_main[i]))

    def save_model_weights(self, step, model_save_path):
        # Helper function to save model weights.
        self.saver.save(self.tf_sess, global_step=step, save_path=model_save_path + config.exp_name)
        print('Model saved to {0}'.format(model_save_path))

    def load_model_weights(self, model_load_path):
        # Helper funciton to load model weights.
        self.saver.restore(self.tf_sess, model_load_path)
        print('Model loaded from {0}'.format(model_load_path))


class Replay_Memory():

    def __init__(self, state_size, memory_size=config.replay_memory_size):
        # The memory essentially stores transitions recorded from the agent
        # taking actions in the environment.
        self.memory_size = memory_size

        # Numpy arrays used as the replay buffer
        if config.extractor_type == 'conv':
            self.states = np.empty((self.memory_size, 84, 84, 4), dtype=np.uint8)
            self.next_states = np.empty((self.memory_size, 84, 84, 4), dtype=np.uint8)
        else:
            self.states = np.empty((self.memory_size, state_size), dtype=np.float32)
            self.next_states = np.empty((self.memory_size, state_size), dtype=np.float32)
        self.actions = np.empty((self.memory_size,), dtype=np.uint8)
        self.rewards = np.empty((self.memory_size,), dtype=np.int8)

        self.dones = np.empty((self.memory_size,), dtype=np.bool)
        self.place_location = 0
        self.filled = False

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # Sample batch_size entries from each of the five arrays
        if self.filled:
            batch = np.random.choice(self.memory_size, batch_size)
        else:
            batch = np.random.choice(self.place_location, batch_size)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        next_states = self.next_states[batch]
        dones = self.dones[batch]

        return states.astype(np.float32), actions, rewards, next_states.astype(np.float32), dones.astype(int)

    def append(self, transition):
        # Appends transition to the memory.
        self.states[self.place_location] = transition[0]
        self.actions[self.place_location] = transition[1]
        self.rewards[self.place_location] = transition[2]
        self.next_states[self.place_location] = transition[3]
        self.dones[self.place_location] = transition[4]
        self.place_location += 1
        if self.place_location == self.memory_size:
            self.place_location = 0
            self.filled = True


class DQN_Agent():
    # In this class, we implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name):
        # Create an environment, instance of the network , as well as the memory.
        self.env = gym.make(environment_name)
        if config.train:
            self.replay_memory = Replay_Memory(self.env.observation_space.shape[0])
        self.model = QNetwork(self.env)
        if config.extractor_type == 'conv':
            self.action_buffer = np.empty((84, 84, 5), dtype=np.float32)

    def epsilon_greedy_policy(self, state, i=0):
        # Creating epsilon greedy probabilities to sample from.
        if config.extractor_type == 'conv':
            self.action_buffer[:, :, 1:5] = self.action_buffer[:, :, 0:4]
            self.action_buffer[:, :, 0] = state
            state = self.action_buffer[:, :, 0:4]
            state = state[np.newaxis]

        Q = self.model.tf_sess.run(self.model.Q, feed_dict={self.model.states: state})

        # Set epsilon
        if i >= config.final_exploration_frame:
            exploration_prob = config.final_exploration
        else:
            exploration_prob = config.initial_exploration + (config.exploration_change_rate) * i

        # Choose an action
        if random.random() < exploration_prob:
            action = random.randrange(0, self.env.env.action_space.n)
        else:
            action = np.argmax(Q)

        next_state, reward, done, _ = self.env.step(action)

        if config.extractor_type == 'conv':
            self.action_buffer[:, :, 1:5] = self.action_buffer[:, :, 0:4]
            self.action_buffer[:, :, 0] = config.preprocess(next_state)
            next_state = self.action_buffer[:, :, 0:4]
            next_state = next_state[np.newaxis]
            self.action_buffer[:, :, 0:4] = self.action_buffer[:, :, 1:5]
            if done:
                self.action_buffer = np.empty((84, 84, 5), dtype=np.float32)

        self.replay_memory.append((state, action, reward, next_state, done))
        return action, reward, next_state, done

    def greedy_policy(self, state):
        # Creating greedy policy for test time.
        if config.extractor_type == 'conv':
            self.action_buffer[:, :, 1:5] = self.action_buffer[:, :, 0:4]
            self.action_buffer[:, :, 0] = state
            state = self.action_buffer[:, :, 0:4]
            state = state[np.newaxis]

        Q = self.model.tf_sess.run(self.model.Q, feed_dict={self.model.states: state})

        if random.random() < config.test_exploration:
            action = random.randrange(0, self.env.env.action_space.n)
        else:
            action = np.argmax(Q)

        next_state, reward, done, _ = self.env.step(action)

        if config.extractor_type == 'conv':
            self.action_buffer[:, :, 1:5] = self.action_buffer[:, :, 0:4]
            self.action_buffer[:, :, 0] = config.preprocess(next_state)
            next_state = self.action_buffer[:, :, 0:4]
            next_state = next_state[np.newaxis]
            self.action_buffer[:, :, 0:4] = self.action_buffer[:, :, 1:5]
            if done:
                self.action_buffer = np.empty((84, 84, 5), dtype=np.float32)

        return action, reward, next_state, done

    def train(self, model_save_path, model_load_path=None):
        # In this function, we train our network.
        # If training without experience replay_memory, we interact with the environment and do stochastic updates
        # If we are using a replay memory, we should interact with environment, store these
        # transitions to memory, while updating the model by sampling batches from the memory.
        self.model.tf_sess = tf.Session()
        self.model.saver = tf.train.Saver(max_to_keep=config.max_iterations)
        init = tf.global_variables_initializer()

        # To initialize from a checkpoint
        if model_load_path is None:
            self.model.tf_sess.run(init)
        else:
            self.model.load_model_weights(model_load_path)

        # Get the initial state
        if config.extractor_type == 'conv':
            state = config.preprocess(self.env.reset())
        else:
            state = self.env.reset()

        # Variables for generating statistics
        reachCount = 0
        episode_count = 0
        iter = []
        avg_rewards = []

        # Main loop
        for i in range(config.max_iterations + 1):

            # Print progress
            if i % 10000 == 0:
                print('Iteration: {0}/{1}'.format(i, config.max_iterations))
                print('Epsilon: {0}'.format(np.maximum((config.initial_exploration + (config.exploration_change_rate) * i), config.final_exploration)))

            # Take an epsilon-greedy step
            action, reward, next_state, done = self.epsilon_greedy_policy(state[np.newaxis], i)

            # Reshape reward for linear mountain car
            if done:
                episode_count += 1
                if config.env_name == 'MountainCar-v0':
                    if next_state[0] >= 0.5 and config.extractor_type == 'linear':
                        reachCount += 1
                        #print(reachCount)
                    else:
                        done = not(done)

            if config.render:
                self.env.render()

            # Get the network input
            if config.use_replay:
                states, actions, rewards, next_states, dones = self.replay_memory.sample_batch(config.batch_size)
            else:
                # Hacky conversions to 'batch' for stochastic online training
                states = state[np.newaxis]
                actions = np.array([action])
                rewards = reward
                next_states = next_state[np.newaxis]
                dones = int(done)

            # Generate targets for the batch
            if config.double:
                Q_next_state = self.model.tf_sess.run(self.model.tar_Q, feed_dict={self.model.states: next_states})
            else:
                Q_next_state = self.model.tf_sess.run(self.model.Q, feed_dict={self.model.states: next_states})

            max_Q_next_state = np.max(Q_next_state, axis=1)
            Q_target = rewards + max_Q_next_state * config.discount_factor * (1. - dones)

            # Update network
            _, loss = self.model.tf_sess.run([self.model.optim, self.model.loss],
                                             feed_dict={self.model.actions: actions, self.model.states: states,
                                                        self.model.Q_target: Q_target})

            # Update current state
            if config.extractor_type == 'conv':
                if done:
                    state = config.preprocess(self.env.reset())
                else:
                    state = next_state[0, :, :, 0]
            else:
                if done:
                    state = self.env.reset()
                else:
                    state = next_state

            # Save model
            # For plots
            if i != 0 and i % 10000 == 0:
                if config.double:
                    self.model.update_target()
                self.model.save_model_weights(i, model_save_path)
                iter.append(i)
                avg_reward = 10000./episode_count
                if config.env_name == 'MountainCar-v0':
                    avg_reward -= 200
                avg_rewards.append(avg_reward)
                episode_count = 0

            # For videos
            if i % (config.max_iterations // 3) == 0:
                self.model.save_model_weights(i, model_save_path)

        # Plot the training rewards
        plt.xlabel('iterations')
        plt.ylabel('Approximate reward per episode (train)')
        plt.plot(iter, avg_rewards, 'orange')
        plt.savefig(config.model_path + config.exp_name + 'TrainPlot' + '.png')

    def test_stats(self, model_load_path, ep_count, step):
        # Evaluates the performance of a loaded model over 100 episodes by calculating cumulative rewards and standard deviations.
        #the step argument gives which model to load
        self.model.tf_sess = tf.Session()
        self.model.saver = tf.train.Saver()
        self.model.load_model_weights(model_load_path)
        if config.capture_videos:
            self.env = gym.wrappers.Monitor(self.env, config.model_path +'Videos/' + config.exp_name + '_' + str(step), force=True,video_callable=lambda episode_id: True)
                
        rewards=[]
        # Initialize
        if config.extractor_type == 'conv':
            state = config.preprocess(self.env.reset())
        else:
            state = self.env.reset()
        episodes = 0
        cumulative_reward = 0.
        episodic_reward=0.

        while episodes < ep_count:

            # Run the test policy
            action, reward, next_state, done = self.greedy_policy(state[np.newaxis])
            cumulative_reward += reward
            episodic_reward += reward

            if config.render:
                self.env.render()

            # Update
            if config.extractor_type == 'conv':
                if done:
                    state = config.preprocess(self.env.reset())
                    episodes += 1
                    rewards.append(episodic_reward)
                    episodic_reward=0.
                else:
                    state = next_state[0, :, :, 0]
            else:
                if done:
                    state = self.env.reset()
                    episodes += 1
                else:
                    state = next_state

        # Print performance
        print('Average reward received: {0}'.format(np.mean(rewards)))
        print('Standard deviation: {0}'.format(np.std(rewards)))

    def test_plots(self):
        # Load the models saved during training and evaluate them on 20 episodes each
        # Generates a plot of the average rewards
        iter = []
        avg_rewards = []
        i = 0
        while (i <= config.max_iterations):
            self.model.tf_sess = tf.Session()
            self.model.saver = tf.train.Saver()
            self.model.load_model_weights(config.model_path + config.exp_name + '-' + str(i))

            if config.capture_videos:
                self.env = gym.wrappers.Monitor(self.env, config.model_path +'Videos/' + config.exp_name, force=True,video_callable=lambda episode_id: True)

            # Initialize
            if config.extractor_type == 'conv':
                state = config.preprocess(self.env.reset())
            else:
                state = self.env.reset()
            episodes = 0
            cumulative_reward = 0.
            ep_count = 20

            while episodes < ep_count:

                # Run the test policy
                action, reward, next_state, done = self.greedy_policy(state[np.newaxis])
                cumulative_reward += reward

                if config.render:
                    self.env.render()

                # Update
                if config.extractor_type == 'conv':
                    if done:
                        state = config.preprocess(self.env.reset())
                        episodes += 1
                    else:
                        state = next_state[0, :, :, 0]
                else:
                    if done:
                        state = self.env.reset()
                        episodes += 1
                    else:
                        state = next_state

            avg_reward = cumulative_reward / ep_count

            # Print performance
            print('Average reward received: {0}'.format(cumulative_reward / ep_count))
            iter.append(i)
            avg_rewards.append(avg_reward)
            i = i + 10000

        # Plot the test rewards
        plt.figure()
        plt.xlabel('iterations')
        plt.ylabel('Average reward per episode (test)')
        plt.plot(iter, avg_rewards)
        plt.savefig(config.model_path + config.exp_name + 'PerformancePlot' + '.png')

    def burn_in_memory(self, burn_in=10000):
        # Initialize replay memory with a burn_in number of transitions.
        for i in range(burn_in):
            # Get a starting state
            if config.extractor_type == 'conv':
                state = config.preprocess(self.env.reset())
                self.action_buffer[:, :, 1:5] = self.action_buffer[:, :, 0:4]
                self.action_buffer[:, :, 0] = state
                state = self.action_buffer[:, :, 0:4]
                state = state[np.newaxis]
            else:
                state = self.env.reset()

            # Take a random action
            action = random.randrange(0, self.env.env.action_space.n)
            next_state, reward, done, _ = self.env.step(action)

            # Add it to the replay memory
            if config.extractor_type == 'conv':
                self.action_buffer[:, :, 1:5] = self.action_buffer[:, :, 0:4]
                self.action_buffer[:, :, 0] = config.preprocess(next_state)
                next_state = self.action_buffer[:, :, 0:4]
                next_state = next_state[np.newaxis]
                self.action_buffer = np.empty((84, 84, 5), dtype=np.float32)
            self.replay_memory.append((state, action, reward, next_state, done))


def main():
    # Create an instance of the DQN_Agent class
    agent = DQN_Agent(config.env_name)
    
    if config.train:
        # Burn in the replay buffer
        if config.use_replay:
            agent.burn_in_memory(config.burn_in)
        # Train
        agent.train(config.model_path)

    # Generate test plots (average rewards over 20 episodes)
    # agent.test_plots()

    # Generate test statistics (average over 100 episodes)
    step=int(input("Enter iteration step"))
    agent.test_stats(config.model_path + config.exp_name + '-' + str(step), 100, step)    


if __name__ == '__main__':
    main()
