from multiprocessing import Process, Pipe
import tensorflow as tf
import numpy as np
import core as model
import Environment
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

class Relational_Proximal_Policy_Optimization:
    def __init__(self):
        self.sess = tf.Session()
        self.height, self.width, self.channel = 84, 84, 4
        self.output_size = 3
        self.hidden = [256, 256]
        self.pi_lr = 0.00025
        self.v_lr = 0.00025
        self.ppo_eps = 0.2
        self.epoch = 3
        self.num_worker = 32
        self.n_step = 256
        self.gamma = 0.99
        self.batch_size = 32
        self.lamda = 0.95

        self.x_ph = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channel])
        self.a_ph = tf.placeholder(tf.int32, shape=[None])
        self.adv_ph = tf.placeholder(tf.float32, shape=[None])
        self.target_ph = tf.placeholder(tf.float32, shape=[None])
        self.old_pi_ph = tf.placeholder(tf.float32, shape=[None, self.output_size])

        
        self.pi, self.v, self.attention = model.relational_network(
            x=self.x_ph,
            hidden=self.hidden,
            output_size=self.output_size,
            activation=tf.nn.relu,
            final_activation=tf.nn.softmax
        )

        self.all_phs = [self.x_ph, self.a_ph, self.adv_ph, self.target_ph, self.old_pi_ph]
        self.get_action_ops = [self.pi, self.v, self.attention]

        self.expanded_action = tf.one_hot(self.a_ph, depth=self.output_size, dtype=tf.float32)
        self.selected_prob = tf.reduce_sum(tf.multiply(self.pi, self.expanded_action), axis=1)
        self.selected_old_prob = tf.reduce_sum(tf.multiply(self.old_pi_ph, self.expanded_action), axis=1)
        self.logpi_prob = tf.log(self.selected_prob)
        self.logoldpi_prob = tf.log(self.selected_old_prob)

        self.ratio = tf.exp(self.logpi_prob - self.logoldpi_prob)
        self.min_adv = tf.where(self.adv_ph > 0, (1.0 + self.ppo_eps)*self.adv_ph, (1.0 - self.ppo_eps)*self.adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(self.ratio * self.adv_ph, self.min_adv))
        self.v_loss = tf.reduce_mean((self.target_ph - self.v)**2)

        self.train_pi = tf.train.AdamOptimizer(self.pi_lr).minimize(self.pi_loss)
        self.train_v = tf.train.AdamOptimizer(self.v_lr).minimize(self.v_loss)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def load_model(self, model_path):
        self.saver.restore(self.sess, model_path)

    def save_model(self, model_path):
        self.saver.save(self.sess, model_path)

    def update(self, state, action, target, adv):
        old_pi = self.sess.run(self.pi, feed_dict={self.x_ph: state})
        sample_range = np.arange(len(state))
        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(state) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j : self.batch_size * (j + 1)]
                state_batch = [state[i] for i in sample_idx]
                action_batch = [action[i] for i in sample_idx]
                adv_batch = [adv[i] for i in sample_idx]
                target_batch = [target[i] for i in sample_idx]
                old_pi_batch = [old_pi[i] for i in sample_idx]
                zip_ph = [state_batch, action_batch, adv_batch, target_batch, old_pi_batch]
                inputs = {k:v for k, v in zip(self.all_phs, zip_ph)}
                self.sess.run([self.train_pi, self.train_v], feed_dict=inputs)

    def get_action(self, state):
        a, v, attention = self.sess.run(self.get_action_ops, feed_dict={self.x_ph: [state]})
        action = np.random.choice(self.output_size, p=a[0])
        return action, v[0], attention[0]

    def test(self):
        self.load_model('model/model')
        works, parent_conns, child_conns = [], [], []
        for idx in range(1):
            parent_conn, child_conn = Pipe()
            work = Environment.Environment(True, idx, child_conn)
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)

        states = np.zeros([self.num_worker, self.height, self.width, self.channel])
        episode_score = 0
        episode = 0

        while True:
            values_list, states_list, actions_list, dones_list, rewards_list = [], [], [], [], []
            for _ in range(self.n_step):
                inf = [self.get_action(s) for s in states]
                actions = [i[0] for i in inf]
                values = [i[1] for i in inf]
                attention = [i[2] for i in inf]

                attention = np.stack(attention)[0]
                attention = np.sum(attention, axis=0)
                attention = np.reshape(attention, [9, 9])
                plt.imshow(attention, cmap='gray')
                plt.show(block=False)
                plt.pause(0.001)
                plt.clf()


                for parent_conn, action in zip(parent_conns, actions):
                    parent_conn.send(action)

                next_states, rewards, dones, real_dones = [], [], [], []
                for parent_conn in parent_conns:
                    s, r, d, rd = parent_conn.recv()
                    next_states.append(s)
                    rewards.append(r)
                    dones.append(d)
                    real_dones.append(rd)

                states_list.append(states)
                values_list.append(values)
                actions_list.append(actions)
                dones_list.append(dones)
                rewards_list.append(rewards)

                episode_score += rewards[0]

                states = next_states


    def run(self):
        works, parent_conns, child_conns = [], [], []
        for idx in range(self.num_worker):
            parent_conn, child_conn = Pipe()
            work = Environment.Environment(True if idx == 0 else False, idx, child_conn)
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)

        states = np.zeros([self.num_worker, self.height, self.width, self.channel])
        episode_score = 0
        episode = 0
        writer = SummaryWriter()

        while True:
            values_list, states_list, actions_list, dones_list, rewards_list = [], [], [], [], []
            for _ in range(self.n_step):
                inf = [self.get_action(s) for s in states]
                actions = [i[0] for i in inf]
                values = [i[1] for i in inf]

                for parent_conn, action in zip(parent_conns, actions):
                    parent_conn.send(action)

                next_states, rewards, dones, real_dones = [], [], [], []
                for parent_conn in parent_conns:
                    s, r, d, rd = parent_conn.recv()
                    next_states.append(s)
                    rewards.append(r)
                    dones.append(d)
                    real_dones.append(rd)

                states_list.append(states)
                values_list.append(values)
                actions_list.append(actions)
                dones_list.append(dones)
                rewards_list.append(rewards)

                episode_score += rewards[0]
                if real_dones[0]:
                    episode += 1
                    writer.add_scalar('data/reward', episode_score, episode)
                    print('episode :', episode, '| episode score :', episode_score)
                    episode_score = 0

                states = next_states

            inf = [self.get_action(s) for s in states]
            actions = [i[0] for i in inf]
            values = [i[1] for i in inf]
            values_list.append(values)

            values_list = np.stack(values_list).transpose([1, 0])

            current_value_list = values_list[:, :-1]
            next_value_list = values_list[:, 1:]

            states_list = np.stack(states_list).transpose([1, 0, 2, 3, 4]).reshape([-1, self.height, self.width, self.channel])
            actions_list = np.stack(actions_list).transpose([1, 0]).reshape([-1])
            dones_list = np.stack(dones_list).transpose([1, 0]).reshape([-1])
            rewards_list = np.stack(rewards_list).transpose([1, 0]).reshape([-1])
            current_value_list = np.stack(current_value_list).reshape([-1])
            next_value_list = np.stack(next_value_list).reshape([-1])

            target_list, adv_list = [], []
            for idx in range(self.num_worker):
                start_idx = idx * self.n_step
                end_idx = (idx + 1) * self.n_step
                adv, target = model.get_gaes(
                    rewards_list[start_idx : end_idx],
                    dones_list[start_idx : end_idx],
                    current_value_list[start_idx : end_idx],
                    next_value_list[start_idx : end_idx],
                    self.gamma,
                    self.lamda,
                    True
                )
                adv_list.append(adv)
                target_list.append(target)
            adv_list = np.stack(adv_list).reshape([-1])
            target_list = np.stack(target_list).reshape([-1])

            self.update(states_list, actions_list, target_list, adv_list)
            self.save_model('model/model')

if __name__ == '__main__':
    ppo = Relational_Proximal_Policy_Optimization()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--disc', required=True, type=str)

    args = parser.parse_args()
    print(args.disc)
    if args.disc == 'train':
        ppo.run()
    elif args.disc == 'test':
        ppo.test()