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

def CNN(x):
    l1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
    l2 = tf.layers.conv2d(inputs=l1, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
    l3 = tf.layers.conv2d(inputs=l2, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)
    shape = l3.get_shape()
    return l3, [s.value for s in shape]

def flatten(nnk, shape):
    flatten = tf.reshape(nnk, [-1, shape[1]*shape[2]*shape[3]])
    return flatten

def query_key_value(nnk, shape):
    flatten = tf.reshape(nnk, [-1, shape[1]*shape[2], shape[3]])
    return flatten, flatten, flatten

def self_attention(query, key, value):
    key_dim_size = float(key.get_shape().as_list()[-1])
    key = tf.transpose(key, perm=[0, 2, 1])
    QK_D = tf.matmul(query, key) / tf.sqrt(key_dim_size)
    attention_map = tf.nn.softmax(QK_D)
    return tf.matmul(attention_map, value), QK_D

def f_theta(self_attention):
    f1 = tf.layers.dense(inputs=self_attention, units=256, activation=tf.nn.relu)
    f2 = tf.layers.dense(inputs=f1, units=256, activation=tf.nn.relu)
    f3 = tf.layers.dense(inputs=f2, units=1, activation=tf.nn.relu)
    map_size_feature = tf.squeeze(f3, axis=2)
    return map_size_feature

def output_layer(f_theta, hidden, output_size, activation, final_activation):
    for h in hidden:
        f_theta = tf.layers.dense(inputs=f_theta, units=h, activation=activation)
    return tf.layers.dense(inputs=f_theta, units=output_size, activation=final_activation)

def relational_network(x, hidden, output_size, activation, final_activation):
    nnk, shape = CNN(x)
    query, key, value = query_key_value(nnk, shape)
    attention, QK_D = self_attention(query, key, value)
    f = f_theta(attention)
    action = output_layer(f, hidden, output_size, activation, final_activation)
    critic = tf.squeeze(output_layer(f, hidden, 1, activation, None), axis=1)
    return action, critic, QK_D

def network(x, hidden, output_size, activation, final_activation):
    nnk, shape = CNN(x)
    flat = flatten(nnk, shape)
    action = output_layer(flat, hidden, output_size, activation, final_activation)
    critic = tf.squeeze(output_layer(flat, hidden, 1, activation, None), axis=1)
    return action, critic

if __name__ == '__main__':
    inp = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
    output = relational_network(inp, [400, 300], 3, tf.nn.relu, tf.nn.softmax)
    output = network(inp, [400, 300], 3, tf.nn.relu, tf.nn.softmax)