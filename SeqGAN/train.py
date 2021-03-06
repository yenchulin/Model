from SeqGAN.models import GeneratorPretraining, DiscriminatorSentence, Generator
from SeqGAN.utils import GeneratorPretrainingGenerator, DiscriminatorSentenceGenerator, Vocab, plotLineChart
from SeqGAN.rl import Agent, Environment
from keras.optimizers import Adam
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
        self.path_pos_sentence = os.path.join(self.top, 'data', 'kokoro_parsed_sentence.txt')
        self.path_neg = os.path.join(self.top, 'data', 'save', 'generated_sentences.txt')
        self.vocab = Vocab(self.path_pos)
        self.g_data = GeneratorPretrainingGenerator(
            path=self.path_pos,
            B=B,
            T=T,
            N=N,
            vocab=self.vocab)
        if os.path.exists(self.path_neg):
            self.d_data = DiscriminatorSentenceGenerator(
                path_pos=self.path_pos_sentence,
                path_neg=self.path_neg,
                B=B,
                T=T,
                N=N,
                vocab=self.vocab)
        self.V = self.vocab.V
        self.agent = Agent(sess, B, self.N, self.V, g_E, g_H, g_lr)
        self.g_beta = Agent(sess, B, self.N, self.V, g_E, g_H, g_lr)
        self.discriminator_sentence = DiscriminatorSentence(self.V, d_dropout)
        self.env = Environment(self.discriminator_sentence.model, self.g_data, self.g_beta, n_sample=n_sample)

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
        self.generator_pre.model.compile(g_adam, 'categorical_crossentropy', sample_weight_mode="temporal")
        print('Generator pre-training')
        self.generator_pre.model.summary()
        
        self.generator_pre.train_on_batch(self.g_data, g_epochs)
        self.generator_pre.model.save_weights(self.g_pre_path)
        self.reflect_pre_train()

    def pre_train_discriminator(self, d_epochs, d_pre_path, lr):
        self.d_pre_path = d_pre_path

        print('Start Generating sentences')
        self.agent.generator.generate_samples(self.T, self.g_data, self.generate_samples, self.path_neg) # agent.generator weights are set after pretraining generator

        self.d_data = DiscriminatorSentenceGenerator(
            path_pos=self.path_pos_sentence,
            path_neg=self.path_neg,
            B=self.B,
            T=self.T,
            N=self.N,
            vocab=self.vocab)

        d_adam = Adam(lr)
        self.discriminator_sentence.model.compile(d_adam, 'binary_crossentropy')
        self.discriminator_sentence.model.summary()
        print('Discriminator pre-training')

        self.discriminator_sentence.train_on_batch(self.d_data, d_epochs)
        self.discriminator_sentence.model.save(self.d_pre_path)

    def load_pre_train(self, g_pre_path, d_pre_path):
        self.generator_pre.model.load_weights(g_pre_path)
        self.reflect_pre_train()
        self.discriminator_sentence.model.load_weights(d_pre_path)

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
        head=1):
        d_adam = Adam(self.d_lr)
        self.discriminator_sentence.model.compile(d_adam, 'binary_crossentropy')
        self.eps = self.init_eps

        print("Adversarial training")
        step_loss_d = []
        step_loss_g = []
        rewards = []

        for step in range(steps):
            # Generator training
            for _ in range(g_steps):
                self.agent.reset() # set agent.generator LSTM h, c state to zero vectorss
                self.env.reset_paragraph()
                for t in range(self.T):
                    self.env.reset_sentence()
                    state = self.env.get_previous_sentence(t) # previous sentence (B, N)
                    g_loss = 0
                    reward_verbose = 0
                    for n in range(self.N):
                        action = self.agent.act(state, epsilon=0.0)  # a word (B, 1) ex. [[20], [2239], [word id]...] or [[0], [0], [0]...] if is the end of sentence
                        reward = self.env.step(action, t, n) # (B, 1)
                        g_loss += self.agent.generator.update(state, action, reward) / self.N # Policy gradient, update generator LSTM h, c, parameters, calulate loss for tha whole sentence
                        reward_verbose += np.mean(reward.reshape(self.B)) / self.N # mean for the batch and each word in the sentence
                    step_loss_g.append(g_loss)
                    rewards.append(reward_verbose)

            # Discriminator training
            for _ in range(d_steps):
                self.agent.generator.generate_samples(
                    self.T,
                    self.g_data,
                    self.generate_samples,
                    self.path_neg)
                self.d_data = DiscriminatorSentenceGenerator(
                    path_pos=self.path_pos_sentence,
                    path_neg=self.path_neg,
                    B=self.B,
                    T=self.T,
                    N=self.N,
                    vocab=self.vocab)
                d_epoch_loss = self.discriminator_sentence.train_on_batch(self.d_data, d_epochs) # shape = (d_epochs, )
                step_loss_d.append(d_epoch_loss)

            # Reflect the weight of agent to env.g_beta (DDQN)
            self.agent.save(g_weights_path) # agent is responds for Reinforcement Learning, acting on state
            self.g_beta.load(g_weights_path) # g_beta is responds for Rollout Policy (Monte Carol Search..)

            self.discriminator_sentence.model.save(d_weights_path)
            self.eps = max(self.eps*(1- float(step) / steps * 4), 1e-4)
        
        # Plot generator loss (a loss for each sentence)
        step_loss_g = np.array(step_loss_g) # (steps * g_steps * T, )
        xlabelName, ylabelName = "Steps", "G Loss"
        top = os.getcwd()
        images = os.path.join(top, 'data', 'save')
        figname = os.path.join(images, 'generator_loss.png')
        plotLineChart(range(1, steps * g_steps * self.T + 1), step_loss_g, xlabelName, ylabelName, figname)

        # Plot discriminator loss
        step_loss_d = np.array(step_loss_d) # (steps * d_steps, d_epochs, )
        xlabelName, ylabelName = "Steps", "D Loss"
        top = os.getcwd()
        images = os.path.join(top, 'data', 'save')
        figname = os.path.join(images, 'discriminator_loss.png')
        plotLineChart(range(1, steps * d_steps * d_epochs + 1), step_loss_d.reshape(steps * d_steps * d_epochs,), xlabelName, ylabelName, figname)

        # Plot reward
        rewards = np.array(rewards) # (steps * g_steps * T, )
        xlabelName, ylabelName = "Steps", "Rewards"
        top = os.getcwd()
        images = os.path.join(top, 'data', 'save')
        figname = os.path.join(images, 'rewards.png')
        plotLineChart(range(1, steps * g_steps * self.T + 1), rewards, xlabelName, ylabelName, figname)


    def save(self, g_path, d_path):
        self.agent.save(g_path)
        self.discriminator_sentence.model.save(d_path)

    def load(self, g_path, d_path):
        self.agent.load(g_path)
        self.g_beta.load(g_path)
        self.discriminator_sentence.model.load_weights(d_path)

    def test(self):
        x, y = self.d_data.next()
        pred = self.discriminator_sentence.model.predict(x)
        for i in range(self.B):
            txt = [self.vocab.id2word[id] for id in x[i].tolist()]
            label = y[i]
            print('{}, {:.3f}: {}'.format(label, pred[i,0], ' '.join(txt)))
