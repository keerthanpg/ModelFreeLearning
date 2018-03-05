#!/usr/bin/env python
#import keras, 
import tensorflow as tf, numpy as np, gym, sys, copy, random
import config
from gym.wrappers import monitor
import matplotlib.pyplot as plt
import pickle

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
		self.hidden1 = config.fc_layer('hidden1', self.states, input_size=num_observations, num_units=10)
		self.hidden2 = config.fc_layer('hidden2', self.hidden1, input_size=10, num_units=20)
		self.hidden3 = config.fc_layer('hidden3', self.hidden2, input_size=20, num_units=30)
		self.Q = config.fc_layer('Q', self.hidden3, input_size=30, num_units=num_actions, activation=None)

		# For dueling networks
		# value = config.fc_layer('value', self.hidden, input_size=100, num_units=1, activation=None)
		# advantage = config.fc_layer('advantage', self.hidden, input_size=100, num_units=num_actions, activation=None)
		# self.Q = (advantage - tf.reshape(tf.reduce_mean(advantage, axis=1), (-1, 1))) + tf.reshape(value, (-1, 1))

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

	def __init__(self, memory_size=50000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions.
		self.memory_size = memory_size
		self.memory_list = [None] * self.memory_size
		self.place_location = 0
		self.filled = False

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.

		# Sample batch_size entries and split them into five arrays
		if self.filled:
			batch = np.array(random.sample(self.memory_list, batch_size)).T
		else:
			batch = np.array(random.sample(self.memory_list[:self.place_location], batch_size)).T

		states = batch[0]
		actions = batch[1]
		rewards = batch[2]
		next_states = batch[3]
		dones = batch[4]

		return states, actions, rewards, next_states, dones

	def append(self, transition):
		# Appends transition to the memory.
		self.memory_list[self.place_location] = transition
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
	
	def __init__(self, environment_name, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		self.env = gym.make(environment_name)
		self.replay_memory = Replay_Memory()
		self.model = QNetwork(self.env)
		# self.plot_avg=[]

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

	def train(self, model_save_path, iters, avg_rewards, model_load_path=None):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		self.model.tf_sess = tf.Session()
		self.model.saver = tf.train.Saver()
		init = tf.global_variables_initializer()

		if model_load_path is None:
			self.model.tf_sess.run(init)
		else:
			self.model.load_model_weights(model_load_path)

		# Get the initial state
		state = self.env.reset()

		for i in range(config.max_iterations):

			# Print progress
			if i % 1000 == 0:
				print('{0}/{1}'.format(i, config.max_iterations))

			# Convert state into a 'batch' for the network
			state = state[np.newaxis]

			# Take an epsilon-greedy step
			action, reward, next_state, done = self.epsilon_greedy_policy(state, i)

			# For stochastic online training
			states = state
			actions = np.array([action])
			rewards = reward
			next_states = next_state[np.newaxis]
			dones = int(done)

			# For experience replay
			# states, actions, rewards, next_states, dones = self.replay_memory.sample_batch(config.batch_size)

			# Generate targets for the batch
			Q_next_state = self.model.tf_sess.run(self.model.Q, feed_dict={self.model.states: next_states})
			max_Q_next_state = np.max(Q_next_state, axis=1)
			Q_target = rewards + max_Q_next_state * config.discount_factor * (1. - dones)

			# For stochastic online training
			Q_target = np.array(Q_target)

			# Update network
			_, loss = self.model.tf_sess.run([self.model.optim, self.model.loss],feed_dict = {self.model.actions: actions, self.model.states: states, self.model.Q_target: Q_target})

			# Update current state
			if done:
				state = self.env.reset()
			else:
				state = next_state

			# Save model
			if i % (config.max_iterations // 3)  == 0:
				self.model.save_model_weights(i, model_save_path)

			if i%10000== 0:				
				ep_count=20
				stateT = self.env.reset()
				episodesT = 0
				cumulative_rewardT = 0.

				while episodesT < ep_count:
					# Convert state to 'batch'
					state = state[np.newaxis]			
					
					# Run the test policy
					actionT, rewardT, next_stateT, doneT = self.epsilon_greedy_policy(state)
					cumulative_rewardT += rewardT

					# Update
					if doneT:
						stateT = self.env.reset()
						episodesT += 1
					else:
						stateT = next_stateT

				# Print performance
				avg_reward=cumulative_rewardT/ep_count
				print('Average reward received: {0}'.format(avg_reward))
				
				#self.model.save_model_weights(i, model_save_path)
				#avg_reward=self.calculate_avg_reward(i)
				iters.append(i)
				avg_rewards.append(avg_reward)

		return (iters,avg_rewards)

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

		while episodes < ep_count:
			# Convert state to 'batch'
			state = state[np.newaxis]
			
			#if(render):
			self.env.render()
			# Run the test policy
			action, reward, next_state, done = self.greedy_policy(state)
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

	def calculate_avg_reward(self, iter):
		#this function calculates and returns avg reward over 20 episodes		
		self.model.tf_sess = tf.Session()
		self.model.saver = tf.train.Saver()	
		self.model.load_model_weights(config.model_path + config.exp_name + '-' + str(iter))	

		ep_count=20
		state = self.env.reset()
		episodes = 0
		cumulative_reward = 0.

		while episodes < ep_count:
			# Convert state to 'batch'
			state = state[np.newaxis]			
			
			# Run the test policy
			action, reward, next_state, done = self.epsilon_greedy_policy(state)
			cumulative_reward += reward

			# Update
			if done:
				state = self.env.reset()
				episodes += 1
			else:
				state = next_state

		# Print performance
		avg_reward=cumulative_reward/ep_count
		print('Average reward received: {0}'.format(avg_reward))

		return(avg_reward)

def generate_plot(iters, avg_rewards):
	plt.xlabel('iterations')
	plt.ylabel('Avg Reward')
	plt.plot(iters, avg_rewards)
	plt.savefig( config.model_path + config.exp_name +'PerformancePlot'+'.png')
	f=open('iters.p', 'wb')
	pickle.dump(iters, f)
	f=open('avg_rewards.p', 'wb')
	pickle.dump(avg_rewards, f)

def main():

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	# gpu_ops = tf.GPUOptions(allow_growth=True)
	# config = tf.ConfigProto(gpu_options=gpu_ops)
	# sess = tf.Session(config=config)

	# You want to create an instance of the DQN_Agent class here, and then train / test it.

	#these variables are used to generate the performance plot
	iters=[]
	avg_rewards=[]

	agent = DQN_Agent(config.env_name, config.render)	
	iters, avg_rewards = agent.train(config.model_path, iters, avg_rewards)
	generate_plot(iters,avg_rewards)	
	agent.test(config.model_path + config.exp_name + '-' + str(int((config.max_iterations-1))), 100)

	#generating videos et al



if __name__ == '__main__':
	main()

