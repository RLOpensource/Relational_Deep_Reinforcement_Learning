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
    x = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    shape = x.get_shape()
    return x, [s.value for s in shape]

def CNN(x):
    l1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    l2 = tf.layers.conv2d(inputs=l1, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    l3 = tf.layers.conv2d(inputs=l2, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    shape = l3.get_shape()
    return l3, [s.value for s in shape]

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
    QK_D = tf.matmul(query, key) / tf.sqrt(key_dim_size)
    attention_map = tf.nn.softmax(QK_D)
    A = tf.matmul(attention_map, value)
    shape = A.get_shape()
    return A, attention_map, [s.value for s in shape]

def output_layer(f_theta, hidden, output_size, activation, final_activation):
    for h in hidden:
        f_theta = tf.layers.dense(inputs=f_theta, units=h, activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return tf.layers.dense(inputs=f_theta, units=output_size, activation=final_activation)

def network(x, hidden, output_size, activation, final_activation):
    nnk, shape = CNN(x)
    flat = flatten(nnk, shape)
    action = output_layer(flat, hidden, output_size, activation, final_activation)
    critic = tf.squeeze(output_layer(flat, hidden, 1, activation, None), axis=1)
    return action, critic, action

def residual_layer_normalization(x, inp, shape, residual_time):
    for i in range(residual_time):
        x = tf.layers.dense(inputs=x, units=shape[2], activation=tf.nn.relu)
        x = x + inp
        mean, variance = tf.nn.moments(x, [2], keep_dims=True)
        x = (x - mean) / tf.sqrt(variance + 1e-8)
    return x

def feature_wise_max(x):
    return tf.reduce_max(x, axis=2)

def relational_network(x, hidden, output_size, activation, final_activation):
    nnk, shape = attention_CNN(x)
    query, key, value, E = query_key_value(nnk, shape)
    A, attention,shape = self_attention(query, key, value)
    normalized_residual_x = residual_layer_normalization(A, E, shape, 2)
    E_hat = feature_wise_max(normalized_residual_x)
    E_hat = tf.layers.dense(inputs=E_hat, units=256, activation=tf.nn.relu)
    actor = output_layer(E_hat, hidden, output_size, tf.nn.relu, tf.nn.softmax)
    critic = tf.squeeze(output_layer(E_hat, hidden, 1, tf.nn.relu, None), axis=1)
    return actor, critic, attention

if __name__ == '__main__':
    inp = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
    nnk, shape = attention_CNN(inp)
    query, key, value, E = query_key_value(nnk, shape)
    A, attention,shape = self_attention(query, key, value)
    normalized_residual_x = residual_layer_normalization(A, E, shape, 3)
    E_hat = feature_wise_max(normalized_residual_x)
    actor = output_layer(E_hat, [256], 3, tf.nn.relu, tf.nn.softmax)
    critic = tf.squeeze(output_layer(E_hat, [256], 1, tf.nn.relu, None), axis=1)

    '''
    query, key, value = query_key_value(nnk, shape)
    A, attention, shape = self_attention(query, key, value)
    normalized_residual_x = residual_layer_normalization(A, shape, 2)
    E_hat = feature_wise_max(normalized_residual_x)
    actor = output_layer(E_hat, [256], 3, tf.nn.relu, tf.nn.softmax)
    critic = tf.squeeze(output_layer(E_hat, [256], 1, tf.nn.relu, None), axis=1)
    '''