from SeqGAN.models import GeneratorPretraining, Discriminator, Generator
from SeqGAN.utils import GeneratorPretrainingGenerator, DiscriminatorGenerator
from SeqGAN.rl import Agent, Environment
from keras.optimizers import Adam
from tqdm import trange
import os
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras.backend as K
K.set_session(sess)

class Trainer(object):
    '''
    Manage training
    '''
    def __init__(self, B, T, N, g_E, g_H, d_E, d_H, d_dropout, g_lr=1e-3, d_lr=1e-3,
        n_sample=16, generate_samples=10000, init_eps=0.1):
        self.B, self.T, self.N = B, T, N
        self.g_E, self.g_H = g_E, g_H
        self.d_E, self.d_H = d_E, d_H
        self.d_dropout = d_dropout
        self.generate_samples = generate_samples
        self.g_lr, self.d_lr = g_lr, d_lr
        self.eps = init_eps
        self.init_eps = init_eps
        self.top = os.getcwd()
        self.path_pos = os.path.join(self.top, 'data', 'kokoro_parsed.txt')
        self.path_neg = os.path.join(self.top, 'data', 'save', 'generated_sentences.txt')
        self.g_data = GeneratorPretrainingGenerator(
            self.path_pos,
            B=B,
            T=T,
            N=N)
        if os.path.exists(self.path_neg):
            self.d_data = DiscriminatorGenerator(
                path_pos=self.path_pos,
                path_neg=self.path_neg,
                B=B,
                T=T,
                N=N)
        self.V = self.g_data.V
        self.agent = Agent(sess, B, self.V, g_E, g_H, g_lr)
        self.g_beta = Agent(sess, B, self.V, g_E, g_H, g_lr)
        self.discriminator = Discriminator(self.V, d_E, d_H, d_dropout)
        self.env = Environment(self.discriminator, self.g_data, self.g_beta, n_sample=n_sample)

        self.generator_pre = GeneratorPretraining(self.V, T, N, g_E, g_H)
        self.g_data.model_s = self.generator_pre.model_1
        self.g_data.model_w = self.generator_pre.model_2
        self.g_data.graph = tf.get_default_graph()

    def pre_train(self, g_epochs, d_epochs, g_pre_path ,d_pre_path, g_lr=1e-3, d_lr=1e-3):
        self.pre_train_generator(g_epochs=g_epochs, g_pre_path=g_pre_path, lr=g_lr)
        self.pre_train_discriminator(d_epochs=d_epochs, d_pre_path=d_pre_path, lr=d_lr)

    def pre_train_generator(self, g_epochs, g_pre_path, lr):
        self.g_pre_path = g_pre_path

        g_adam = Adam(lr)
        self.generator_pre.model.compile(g_adam, 'categorical_crossentropy')
        print('Generator pre-training')
        self.generator_pre.model.summary()
        
        # Pretrain
        for _ in trange(g_epochs, ascii=True):
            for _ in range(self.g_data.__len__()): # Total number of steps (number of batches = num_samples / batch_size)
                self.generator_pre.model.train_on_batch(
                    x=self.g_data.next()[0],
                    y=self.g_data.next()[1]
                )

        self.generator_pre.model.save_weights(self.g_pre_path)
        self.reflect_pre_train()

    def pre_train_discriminator(self, d_epochs, d_pre_path, lr):
        self.d_pre_path = d_pre_path

        print('Start Generating sentences')
        self.agent.generator.generate_samples(self.T, self.g_data, self.generate_samples, self.path_neg) # agent.generator weights are set after pretraining generator

        self.d_data = DiscriminatorGenerator(
            path_pos=self.path_pos,
            path_neg=self.path_neg,
            B=self.B,
            T=self.T,
            N=self.N,
            shuffle=True)

        d_adam = Adam(lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.discriminator.summary()
        print('Discriminator pre-training')

        self.discriminator.fit_generator(
            self.d_data,
            steps_per_epoch=None,
            epochs=d_epochs)
        self.discriminator.save(self.d_pre_path)

    def load_pre_train(self, g_pre_path, d_pre_path):
        self.generator_pre.model.load_weights(g_pre_path)
        self.reflect_pre_train()
        self.discriminator.load_weights(d_pre_path)

    def load_pre_train_g(self, g_pre_path):
        self.generator_pre.model.load_weights(g_pre_path)
        self.reflect_pre_train()

    def load_pre_train_d(self, d_pre_path):
        self.discriminator.load_weights(d_pre_path)


    def reflect_pre_train(self):
        i = 0
        for layer in self.generator_pre.model_1.layers + self.generator_pre.model_2.layers + self.generator_pre.model_3.layers:
            if len(layer.get_weights()) != 0:
                w = layer.get_weights()
                self.agent.generator.layers[i].set_weights(w)
                self.g_beta.generator.layers[i].set_weights(w)
                i += 1

    def train(self, steps=10, g_steps=1, d_steps=1, d_epochs=1,
        g_weights_path='data/save/generator.pkl',
        d_weights_path='data/save/discriminator.hdf5',
        verbose=True,
        head=1):
        d_adam = Adam(self.d_lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.eps = self.init_eps
        for step in range(steps):
            # Generator training
            for _ in range(g_steps):
                rewards = np.zeros([self.B, self.T])
                self.agent.reset() # set agent.generator LSTM h, c state to zero vectorss
                self.env.reset()
                for t in range(self.T):
                    state = self.env.get_state() # ex. t = 0, env.t = 1, (B, 1) [[1], [1], [1], ...]
                    action = self.agent.act(state, epsilon=0.0)  # (B, 1) ex. [[20], [2239], [word id]...] or [[0], [0], [0]...] if is end sentence
                    next_state, reward, is_episode_end, info = self.env.step(action)
                    self.agent.generator.update(state, action, reward) # Policy gradient, update generator LSTM h, c, parameters
                    rewards[:, t] = reward.reshape([self.B, ])
                    if is_episode_end:
                        if verbose:
                            print('Reward: {:.3f}, Episode end'.format(np.average(rewards)))
                            self.env.render(head=head)
                        break
            # Discriminator training
            for _ in range(d_steps):
                self.agent.generator.generate_samples(
                    self.T,
                    self.g_data,
                    self.generate_samples,
                    self.path_neg)
                self.d_data = DiscriminatorGenerator(
                    path_pos=self.path_pos,
                    path_neg=self.path_neg,
                    B=self.B,
                    T=self.T,
                    N=self.N,
                    shuffle=True)
                self.discriminator.fit_generator(
                    self.d_data,
                    steps_per_epoch=None,
                    epochs=d_epochs)

            # Update env.g_beta to agent
            self.agent.save(g_weights_path) # agent is responds for Reinforcement Learning, acting on state
            self.g_beta.load(g_weights_path) # g_beta is responds for Rollout Policy (Monte Carol Search..)

            self.discriminator.save(d_weights_path)
            self.eps = max(self.eps*(1- float(step) / steps * 4), 1e-4)

    def save(self, g_path, d_path):
        self.agent.save(g_path)
        self.discriminator.save(d_path)

    def load(self, g_path, d_path):
        self.agent.load(g_path)
        self.g_beta.load(g_path)
        self.discriminator.load_weights(d_path)

    def test(self):
        x, y = self.d_data.next()
        pred = self.discriminator.predict(x)
        for i in range(self.B):
            txt = [self.g_data.id2word[id] for id in x[i].tolist()]
            label = y[i]
            print('{}, {:.3f}: {}'.format(label, pred[i,0], ' '.join(txt)))
