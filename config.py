import tensorflow as tf
import skimage

# Tensorflow FC layer
def fc_layer(name, input, input_size, num_units, activation=tf.nn.relu):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=[input_size, num_units], initializer=kernel_initializer)
        bias = tf.get_variable('bias', shape=[num_units], initializer=bias_initializer)
    if activation is not None:
        return activation(tf.matmul(input, weights) + bias)
    else:
        return tf.matmul(input, weights) + bias

# Tensorflow conv layer (for space invaders)
def conv_layer(name, input, shape, stride, activation=tf.nn.relu):
    with tf.variable_scope(name):
        conv_weights = tf.get_variable('kernel', shape=shape, initializer=kernel_initializer)
        conv_bias = tf.get_variable('bias', shape=[shape[3]], initializer=bias_initializer)
    return activation(tf.nn.conv2d(input, conv_weights, strides=(1, stride, stride, 1), padding='VALID') + conv_bias)

# Process space invaders frames
def preprocess(frame):
    return skimage.color.rgb2gray(skimage.transform.resize(frame, (84,84)))

# Experiment name
exp_name = 'LinearQ-CartPole-1'

# Environment
env_name = 'CartPole-v0'

# Whether to render
render = False

# Whether to train
train = True

# Weights initializer
kernel_initializer = tf.contrib.layers.xavier_initializer()
bias_initializer = tf.constant_initializer(0.0)

# Model directory
model_path = 'tmp/'

# Maximum iterations for training
max_iterations = 1000000

# Gamma
discount_factor = 0.99

# Learning rate
lr = 0.0001

# Epsilon greedy exploration
initial_exploration = 0.5
final_exploration = 0.05
final_exploration_frame = max_iterations
exploration_change_rate = (final_exploration - initial_exploration)/final_exploration_frame
test_exploration = 0.05

# Batch size
batch_size = 32

# Experience replay memory size
replay_memory_size = 50000

# Number of burn in actions
burn_in = 10000