import tensorflow as tf
import numpy as np
import copy

def get_gaes(rewards, dones, values, next_values, gamma, lamda, normalize):
    deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
    deltas = np.stack(deltas)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(deltas) - 1)):
        gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

    target = gaes + values
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    return gaes, target

def attention_CNN(x):
    x = tf.layers.conv2d(inputs=x, filters=12, kernel_size=[3, 3], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=24, kernel_size=[3, 3], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
    
    shape = x.get_shape()
    return x, [s.value for s in shape]

def flatten(nnk, shape):
    flatten = tf.reshape(nnk, [-1, shape[1]*shape[2]*shape[3]])
    return flatten

def query_key_value(nnk, shape):
    flatten = tf.reshape(nnk, [-1, shape[1]*shape[2], shape[3]])
    after_layer = [tf.layers.dense(inputs=flatten, units=shape[3], activation=tf.nn.relu) for i in range(3)]

    return after_layer[0], after_layer[1], after_layer[2], flatten

def self_attention(query, key, value):
    key_dim_size = float(key.get_shape().as_list()[-1])
    key = tf.transpose(key, perm=[0, 2, 1])
    S = tf.matmul(query, key) / tf.sqrt(key_dim_size)
    attention_weight = tf.nn.softmax(S)
    A = tf.matmul(attention_weight, value)
    shape = A.get_shape()
    return A, attention_weight, [s.value for s in shape]

def output_layer(f_theta, hidden, output_size, activation, final_activation):
    for h in hidden:
        f_theta = tf.layers.dense(inputs=f_theta, units=h, activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return tf.layers.dense(inputs=f_theta, units=output_size, activation=final_activation)

def layer_normalization(x):
    feature_shape = x.get_shape()[-1:]
    mean, variance = tf.nn.moments(x, [2], keep_dims=True)
    beta = tf.Variable(tf.zeros(feature_shape), trainable=False)
    gamma = tf.Variable(tf.ones(feature_shape), trainable=False)
    return gamma * (x - mean) / tf.sqrt(variance + 1e-8) + beta

def residual(x, inp, residual_time):
    for i in range(residual_time):
        x = x + inp
        x = layer_normalization(x)
    return x

def feature_wise_max(x):
    return tf.reduce_max(x, axis=2)

def relational_network(x, hidden, output_size, activation, final_activation):
    nnk, shape = attention_CNN(x)
    query, key, value, E = query_key_value(nnk, shape)
    normalized_query = layer_normalization(query)
    normalized_key = layer_normalization(key)
    normalized_value = layer_normalization(value)
    A, attention_weight, shape = self_attention(normalized_query, normalized_key, normalized_value)
    E_hat = residual(A, E, 2)
    max_E_hat = feature_wise_max(E_hat)
    actor = output_layer(max_E_hat, hidden, output_size, activation, final_activation)
    critic = tf.squeeze(output_layer(max_E_hat, hidden, 1, tf.nn.relu, None), axis=1)
    return actor, critic, attention_weight

'''
if __name__ == '__main__':
    inp = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
    nnk, shape = attention_CNN(inp)
    query, key, value, E = query_key_value(nnk, shape)
    normalized_query = layer_normalization(query)
    normalized_key = layer_normalization(key)
    normalized_value = layer_normalization(value)
    A, attention_weight, shape = self_attention(normalized_query, normalized_key, normalized_value)
    E_hat = residual(A, E, 4)
    max_E_hat = feature_wise_max(E_hat)
    actor = output_layer(max_E_hat, [256], 3, tf.nn.relu, tf.nn.softmax)
    critic = tf.squeeze(output_layer(max_E_hat, [256], 1, tf.nn.relu, None). axis=1)
'''