from SeqGAN.models import Generator, GeneratorPretraining, Discriminator
from SeqGAN.utils import DiscriminatorGenerator, Vocab
import keras.backend as K
import numpy as np

class Agent(object):
    '''
    On each step, Agent act on state.
    Then Environment return next state, reward, and so on.
    '''
    def __init__(self, sess, B, V, E, H, lr=1e-3):
        '''
        # Arguments:
            sess: tf.Session
            B: int, batch_size
            V: int, Vocabrary size
            E: int, Embedding size
            H: int, LSTM hidden size
        # Optional Arguments:
            lr: float, learning rate, default is 0.001
        '''
        self.sess = sess
        self.num_actions = V
        self.B = B
        self.V = V
        self.E = E
        self.H = H
        self.lr = lr
        self.eps = 0.1
        self.generator = Generator(sess, B, V, E, H, lr)

    def act(self, state, epsilon=0, deterministic=False):
        '''
        # Arguments:
            state: nparray, dtype = int, previous sentence, shape = (B, N)
            epsilon: float, 0 <= epsilon <= 1,
                if epsilon is 1, the Agent will act completely random.
        # Returns:
            action: nparray, dtype=int, shape = (B, 1)
        '''
        if np.random.rand() <= epsilon:
            action = np.random.randint(low=0, high=self.num_actions, size=(self.B, 1))
        elif not deterministic:
            h_s = self.generator.predict_h_s(state) # (B, 1024)    
            prob = self.generator.predict_word(h_s) # (B, V)
            action = self.generator.sampling_word(prob).reshape(-1, 1) # (B, 1)
        else:
            h_s = self.generator.predict_h_s(state) # (B, 1024) 
            prob = self.generator.predict_word(h_s) # (B, V)
            action = np.argmax(prob, axis=-1).reshape(-1, 1) # (B, 1)

        # TODO: have to know whether this is a last word/sentence or not. (refer to is_end in origin Seqgan)
        return action

    def reset(self):
        self.generator.reset_rnn_state()

    def save(self, path):
        self.generator.save(path)

    def load(self, path):
        self.generator.load(path)


class Environment(object):
    '''
    On each step, Agent act on state.
    Then Environment return next state, reward, and so on.
    '''
    def __init__(self, discriminator, data_generator, g_beta, n_sample=16):
        '''
        Environment class for Reinforced Learning
        # Arguments:
            discriminator: keras model
            data_generator: SeqGAN.models.GeneratorPretrainingGenerator
            g_beta: SeqGAN.rl.Agent, copy of Agent
                params of g_beta.generator should be updated with those of original
                generator on regular occasions.
        # Parameters:
            t: int, an indicator of the previous timestep for sentences
            n: int, an indicator of the previous timestep for words
            _state: nparray, dtype=int, previous generated sentences and words, shape = (B, T, N)
        # Optional Arguments:
            n_sample: int, default is 16, the number of Monte Calro search sample
        '''
        self.data_generator = data_generator
        self.B = data_generator.B
        self.T = data_generator.T
        self.N = data_generator.N
        self.n_sample = n_sample
        self.discriminator = discriminator
        self.g_beta = g_beta
        self.reset_paragraph()
        self.reset_sentence()

    def get_previous_sentence(self, idx):
        """
        Get previous sentence.
        idx: the previous sentence index.
        """
        return self._state_paragraph[:, idx] # (B, N)

    def reset_paragraph(self):
        """
        Should be called when the model is going to generate a new parageaph sample.
        """
        # Create the first (initial) sentence in state
        self._state_paragraph = np.zeros([self.B, 1, self.N], dtype=np.int32) # (B, 1, N)
        self._state_paragraph[:, 0, 0] = Vocab.BOS
        self._state_paragraph[:, 0, 1] = Vocab.EOS # <S> </S> <PAD> <PAD>.....
        self.g_beta.reset() # Agent reset (g_beta.generator LSTM h, c state to zero vectors)

    def reset_sentence(self):
        """
        Should be called when the model is going to generate a new sentence in a paragraph sample.
        """
        self._state_sentence = np.zeros([self.B, 0]) # (B, 0) with zero words
        
    def step(self, action, t, n):
        '''
        Step 1-step forward and calculate the reward of the given Agent action.
        If the action is the last word of the sentence, append the sentence to the paragraph.
        # Arguments:
            action: numpy array, dtype=int, the selected word, shape = (B, 1)
            t: current time step when generating the paragraph, the previous sentence index.
            n: current time step when generating the sentence
        # Returns:
            reward: nparray, dtype=float, shape = (B, 1)
        '''
        is_episode_end = n + 1 >= self.N
        self._append_word(action)
        if is_episode_end:
            self._append_sentence()
        reward = self.Q(n, is_episode_end, self.n_sample) # calculate the reward of the appended action
        return reward

    def render(self, head=1):
        for i in range(head):
            ids = self._state_sentence[i] # (N, )
            words = [self.data_generator.vocab.id2word[_id] for _id in ids.tolist()]
            print(' '.join(words))
        print('-' * 80)


    def Q(self, t, n, is_last_word, n_sample=16):
        '''
        State-Action value function using Rollout policy.
        Calculate the reward of the previous state and action (already appended).
        # Arguments:
            t: current time step when generating the paragraph, the previous sentence index.
            n: current time step when generating the sentence
            is_last_word: bool

        # Optional Arguments:
            n_sample: int, default is 16, number of samples for Monte Calro Search

        # Returns:
            reward: nparray, dtype=float, shape = (B, 1), State-Action value

        # Requires:
            t, T: used to define time range.
            state: determined texts, Y[0:t-1], used for Rollout.
            action: next words, y[t], used for sentence Y[0:t].
            g_beta: Rollout policy.
        '''
        hidden_and_cell_states = self.g_beta.generator.get_rnn_state() # tuple, get the h(s) and c(s) before Monte Carol
        reward = np.zeros([self.B, 1])
        Y = self._state_sentence # (B, n), the sentence that is currently generating words (action appended).

        if is_last_word: # last word, no need to do Monte Carol
            return self.discriminator.predict(Y)

        # Rollout
        for _ in range(n_sample): # first get the whole sentence, and have D predict the score, lastly, repeated n_sample times 
            self.g_beta.generator.set_rnn_state(hidden_and_cell_states) # reset the h, c states to the ones before Monte Carol
            for _ in range(n+1, self.N): # calculate the rest of the words n+1 to N (Monte Carol)
                y_n = self.g_beta.act(self.get_previous_sentence(t), epsilon=self.g_beta.eps)
                Y = self._append_word(y_n, state=Y)
            reward += self.discriminator.predict(Y) / n_sample

        return reward # (B, 1)

    def _append_word(self, word, state=None):
        '''
        Append word to sentence.
        # Arguments:
            word: nparray, dtype=int, shape=(B, 1)
            state: nparray, dtype=int, the sentence that is generating words, shape=(B, n)
        '''
        if state is None:
            self._state_sentence = np.concatenate([self._state_sentence, word], axis=-1)
        else:
            return np.concatenate([state, word], axis=-1)

    def _append_sentence(self):
        '''
        Append the sentence state to paragraph.
        # Arguments:
            sentence: nparray, dtype=int, shape=(B, N)
        '''
        self._state_paragraph = np.concatenate([self._state_paragraph, self._state_sentence.reshape(self.B, 1, self.N)], axis=-2)
