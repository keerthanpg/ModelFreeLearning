import tensorflow as tf

# Tensorflow FC layer
def fc_layer(name, input, input_size, num_units, activation=tf.nn.relu):
    weights = tf.Variable(kernel_initializer([input_size, num_units]), name = name + '_weights')
    bias = tf.Variable(bias_initializer([num_units]), name = name + '_bias')
    if activation is not None:
        return activation(tf.matmul(input, weights) + bias)
    else:
        return tf.matmul(input, weights) + bias

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
final_exploration_frame = 1000000
exploration_change_rate = (final_exploration - initial_exploration)/final_exploration_frame
test_exploration = 0.05

# Batch size
batch_size = 32

# Experience replay memory size
replay_memory_size = 50000

# Number of burn in actions
burn_in = 10000