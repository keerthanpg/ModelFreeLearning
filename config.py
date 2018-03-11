import tensorflow as tf
import skimage.color, skimage.transform

# Hyper-parameters
# Experiment name
exp_name = 'DDQN-SpaceInvaders'

# Environment
env_name = 'SpaceInvaders-v0'

# Whether to train
train = True

# Whether to plot
plot = True

# Whether to get stats
stats = False

# Whether to render
render = False

# Whether to save videos while getting stats
capture_videos = False

# Feature extractor ('linear', 'fc', 'conv')
extractor_type = 'conv'

# Whether to use the experience replay
use_replay = True

# Standard DQN or dueling
dueling = False

# Standard or DDQN
double = True

# Weights initializer
kernel_initializer = tf.contrib.layers.xavier_initializer()
bias_initializer = tf.constant_initializer(0.0)

# Model directory
model_path = 'tmp/'

# Maximum iterations for training
max_iterations = 2000000

# Gamma
discount_factor = 0.99

# Learning rate
lr = 0.0001

# Epsilon greedy exploration
initial_exploration = 1.
final_exploration = 0.1
final_exploration_frame = max_iterations / 2
exploration_change_rate = (final_exploration - initial_exploration) * (1. / final_exploration_frame)
test_exploration = 0.1

# Batch size
batch_size = 32

# Experience replay memory size
replay_memory_size = 250000

# Number of burn in actions
burn_in = 10000


# Helper functions
# Tensorflow FC layer
def fc_layer(name, input, input_size, num_units, activation=tf.nn.relu):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=[input_size, num_units], initializer=kernel_initializer)
        bias = tf.get_variable('bias', shape=[num_units], initializer=bias_initializer)
    if activation is not None:
        return activation(tf.matmul(input, weights) + bias)
    else:
        return tf.matmul(input, weights) + bias


# Tensorflow conv layer for space invaders
def conv_layer(name, input, shape, stride, activation=tf.nn.relu):
    with tf.variable_scope(name):
        conv_weights = tf.get_variable('kernel', shape=shape, initializer=kernel_initializer)
        conv_bias = tf.get_variable('bias', shape=[shape[3]], initializer=bias_initializer)
    return activation(tf.nn.conv2d(input, conv_weights, strides=(1, stride, stride, 1), padding='VALID') + conv_bias)


# Feature extractors
# 'fc' for DQNs
def fc_extractor(input, input_size):
    hidden1 = fc_layer('hidden1', input, input_size=input_size, num_units=30)
    hidden2 = fc_layer('hidden2', hidden1, input_size=30, num_units=30)
    hidden3 = fc_layer('hidden3', hidden2, input_size=30, num_units=30)
    return hidden3


# 'conv' for Space Invaders
def conv_extractor(input):
    normalize = (input - (255.0 / 2)) / (255.0 / 2)
    conv1 = conv_layer('conv1', normalize, shape=[8, 8, 4, 32], stride=4)
    conv2 = conv_layer('conv2', conv1, shape=[4, 4, 32, 64], stride=2)
    conv3 = conv_layer('conv3', conv2, shape=[3, 3, 64, 64], stride=1)
    flatten = tf.reshape(conv3, (-1, 7 * 7 * 64))
    fc = fc_layer('fc', flatten, input_size=7 * 7 * 64, num_units=512)
    return fc


# Choice of extractor
def extractor(input, input_size, type):
    if type == 'linear':
        return input, input_size
    elif type == 'fc':
        return fc_extractor(input, input_size), 30
    else:
        return conv_extractor(input), 512


# Q estimation from features
def estimate_Q(input, input_size, num_actions, dueling=False):
    if dueling:
        value = fc_layer('value', input, input_size=input_size, num_units=1, activation=None)
        advantage = fc_layer('advantage', input, input_size=input_size, num_units=num_actions, activation=None)
        Q = (advantage - tf.reshape(tf.reduce_mean(advantage, axis=1), (-1, 1))) + tf.reshape(value, (-1, 1))
    else:
        Q = fc_layer('Q', input, input_size=input_size, num_units=num_actions, activation=None)
    return Q


# Process space invaders frames
def preprocess(frame):
    return skimage.color.rgb2gray(skimage.transform.resize(frame, (84, 84)))