#!/usr/bin/env python
import tensorflow as tf, numpy as np, gym, sys, copy, random
import config


class QNetwork():
	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, env):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.
		self.tf_sess = None
		self.saver = None
		num_observations = env.observation_space.shape[0]
		num_actions = env.action_space.n

		# Model
		self.states = tf.placeholder(name='states', shape=(None, num_observations), dtype=tf.float32)
		self.features, feat_length = config.extractor(self.states, num_observations, config.extractor_type)
		self.Q = config.estimate_Q(self.features, input_size=feat_length, num_actions=num_actions, dueling=config.dueling)

		# Loss
		self.Q_target = tf.placeholder(name='Q_target', shape=(None,), dtype=tf.float32)
		self.actions = tf.placeholder(name='actions', shape=(None,), dtype=tf.int32)
		actions_one_hot = tf.one_hot(self.actions, depth=num_actions)
		self.Q_pred = tf.reduce_sum(actions_one_hot * self.Q, axis=1)
		self.loss = tf.reduce_mean((self.Q_target - self.Q_pred)**2)

		# Optimizer
		self.optim = tf.train.RMSPropOptimizer(learning_rate=config.lr).minimize(self.loss)

	def save_model_weights(self, step, model_save_path):
		# Helper function to save your model / weights.
		self.saver.save(self.tf_sess, global_step=step, save_path=model_save_path + config.exp_name)
		print('Model saved to {0}'.format(model_save_path))

	def load_model_weights(self, model_load_path):
		# Helper funciton to load model weights.
		#self.tf_sess.run(tf.global_variables_initializer())
		self.saver.restore(self.tf_sess, model_load_path)
		print('Model loaded from {0}'.format(model_load_path))


class Replay_Memory():

	def __init__(self, state_size, memory_size=50000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions.
		self.memory_size = memory_size
		if config.extractor_type == 'conv':
			self.states = np.empty((self.memory_size, 84, 84, 4), dtype=np.float32)
			self.next_states = np.empty((self.memory_size, 84, 84, 4), dtype=np.float32)
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
		# You will feed this to your model to train.

		# Sample batch_size entries and split them into five arrays
		if self.filled:
			batch = np.random.choice(self.memory_size, batch_size)
		else:
			batch = np.random.choice(self.place_location, batch_size)

		states = self.states[batch]
		actions = self.actions[batch]
		rewards = self.rewards[batch]
		next_states = self.next_states[batch]
		dones = self.dones[batch]

		return states, actions, rewards, next_states, dones.astype(int)

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
	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		self.env = gym.make(environment_name)
		self.replay_memory = Replay_Memory(self.env.observation_space.shape[0])
		self.model = QNetwork(self.env)

	def epsilon_greedy_policy(self, state, i=0, test_mode=False):
		# Creating epsilon greedy probabilities to sample from.
		Q = self.model.tf_sess.run(self.model.Q, feed_dict={self.model.states: state})

		# Set epsilon
		if test_mode:
			exploration_prob = config.test_exploration
		else:
			if i >= config.final_exploration_frame:
				exploration_prob = config.final_exploration
			else:
				exploration_prob = config.initial_exploration + i * config.exploration_change_rate

		# Choose an action
		if random.random() < exploration_prob:
			action = random.randrange(0, self.env.env.action_space.n)
		else:
			action = np.argmax(Q)

		next_state, reward, done, _ = self.env.step(action)
		self.replay_memory.append((state, action, reward, next_state, done))
		return action, reward, next_state, done

	def greedy_policy(self, state):
		# Creating greedy policy for test time.
		Q = self.model.tf_sess.run(self.model.Q, feed_dict={self.model.states: state})
		action = np.argmax(Q)
		next_state, reward, done, _ = self.env.step(action)
		return action, reward, next_state, done

	def train(self, model_save_path, model_load_path=None):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		self.model.tf_sess = tf.Session()
		self.model.saver = tf.train.Saver(max_to_keep=config.max_iterations)
		init = tf.global_variables_initializer()

		if model_load_path is None:
			self.model.tf_sess.run(init)
		else:
			self.model.load_model_weights(model_load_path)

		# Get the initial state
		state = self.env.reset()

		for i in range(config.max_iterations+1):

			# Print progress
			if i % 1000 == 0:
				print('{0}/{1}'.format(i, config.max_iterations))


			# Take an epsilon-greedy step
			action, reward, next_state, done = self.epsilon_greedy_policy(state[np.newaxis], i)

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
			Q_next_state = self.model.tf_sess.run(self.model.Q, feed_dict={self.model.states: next_states})
			max_Q_next_state = np.max(Q_next_state, axis=1)
			Q_target = rewards + max_Q_next_state * config.discount_factor * (1. - dones)

			# Update network
			_, loss = self.model.tf_sess.run([self.model.optim, self.model.loss],feed_dict = {self.model.actions: actions, self.model.states: states, self.model.Q_target: Q_target})

			# Update current state
			if done:
				state = self.env.reset()
			else:
				state = next_state

			# Save model
			# For plots
			if i % 100000  == 0:
				self.model.save_model_weights(i, model_save_path)

			# For videos
			if i % (config.max_iterations // 3) == 0:
				self.model.save_model_weights(i, model_save_path)

	def test(self, model_load_path, ep_count):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		self.model.tf_sess = tf.Session()
		self.model.saver = tf.train.Saver()
		self.model.load_model_weights(model_load_path)

		# Initialize
		state = self.env.reset()
		episodes = 0
		cumulative_reward = 0.

		if config.render:
			self.env.render()

		while episodes < ep_count:

			# Run the test policy
			action, reward, next_state, done = self.greedy_policy(state[np.newaxis])
			cumulative_reward += reward

			# Update
			if done:
				state = self.env.reset()
				episodes += 1
			else:
				state = next_state

		# Print performance
		print('Average reward received: {0}'.format(cumulative_reward/ep_count))

	def burn_in_memory(self, burn_in=10000):
		# Initialize your replay memory with a burn_in number of episodes / transitions. 
		for i in range(burn_in):
			state = self.env.reset()
			action = random.randrange(0, self.env.env.action_space.n)
			next_state, reward, done, _ = self.env.step(action)
			self.replay_memory.append((state, action, reward, next_state, done))


def main():

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	# gpu_ops = tf.GPUOptions(allow_growth=True)
	# config = tf.ConfigProto(gpu_options=gpu_ops)
	# sess = tf.Session(config=config)

	# You want to create an instance of the DQN_Agent class here, and then train / test it.

	agent = DQN_Agent(config.env_name)
	if config.use_replay:
		agent.burn_in_memory(config.burn_in)
	agent.train(config.model_path)
	agent.test(config.model_path + config.exp_name + '-' + str(config.max_iterations), 100)


if __name__ == '__main__':
	main()

