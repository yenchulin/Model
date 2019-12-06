import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Dropout, concatenate, Reshape, RepeatVector
from keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from keras.layers import Activation
from keras.layers.wrappers import TimeDistributed
from keras.utils import to_categorical
import tensorflow as tf
import pickle

def visualAttention(x):
    '''
    # Arguments:
        x: hidden state from LSTM, shape (B, 4096), a tensor
    # Variables:
        feature: tensor with shape (1, 50, 4096), 50 visual features with 4096 dim
    # Returns:
        context: tensor with shape (B, 4096), visual context vector
    '''
    feature = [range(4096)] * 50 # TODO: should be replaced
    feature = np.array(feature)
    feature = tf.constant(feature, dtype=tf.float32) # (50, 4096)
    feature = tf.expand_dims(feature, 0) # (1, 50,4096)

    x_expand = tf.reshape(tf.tile(x, [1,tf.shape(feature)[1]]), [tf.shape(x)[0], tf.shape(feature)[1], tf.shape(feature)[2]]) # duplicate 4096-hidden-state for 50 times (B, 50, 4096)
    dot = tf.reduce_sum(tf.multiply(feature, x_expand), -1) # (B, 50)
    weight = tf.nn.softmax(dot)
    weight = tf.expand_dims(weight, -1) # (B, 50, 1)
    # TODO: Normalization
    context = tf.reduce_sum(tf.multiply(weight, feature), -2) # (B, 4096)
    return context

class GeneratorPretraining():
    '''
    Model for Generator pretraining. This model's weights should be shared with
        Generator.
    # Arguments:
        V: int, Vocabrary size
        T: int, Max sentences in a paragraph
        N: int, Max words in a sentence
    # Parameters:
        model_1: 
            Model for training h_s.
            Inputs are data, h_s_pre. 
            Output is h_s.
        model_2: 
            Model for training h_w.
            Inputs are h_s, h_w_pre. 
            Output is h_w.
        model_3: 
            Model for the final output, probability distribution.
            Input is h_w. 
            Output is the probability distribution of a paragraph (B, T, N, V).
        model: 
            The merged model of model_1, model_2, model_3. 
            Inputs are data, h_s_pre, h_w_pre. 
            Output is the probability distribution of a paragraph (B, T, N, V).
    '''
    def __init__(self, V, T, N, E, H):
        self.V = V
        self.T = T
        self.N = N
        self.__build_graph__()

    def __build_graph__(self):
        # in comment, B means batch size, T means lengths of time steps.

        # Model 1 (1st part)
        data = Input(shape=(self.T, self.N), dtype='int32', name='Input') # (B, T, N)
        h_s_pre = Input(shape=(self.T, 1024), dtype='float32', name='SentencePreviousHidden') # (B, T, 1024)
        
        out = TimeDistributed(
            Embedding(self.V, 512, mask_zero=True),  
            name="WordEmbedding")(data) # (B, T, N, 512)
        out = Lambda(lambda x: tf.reduce_mean(x, axis=2), name="SentenceEmbedding")(out) # average word embeddings (B, T, 512
        h_p = LSTM(512, return_sequences=True, name='ParagraphRNN')(out) # (B, T, 512)
        out = concatenate([h_p, h_s_pre]) # (B, T, 512 + 1024)
        out = TimeDistributed(
            Dense(4096),
            name='ExpandDim')(out) # (B, T, 4096)
        out = TimeDistributed(
            Lambda(visualAttention),
            name='VisualAttention')(out) # (B, T, 4096)
        h_s = LSTM(1024, return_sequences=True, name='SentenceRNN')(out) # (B, T, 1024)
        self.model_1 = Model(inputs=[data, h_s_pre], outputs=h_s, name='model_1')

        # Model 2 (2nd part)
        h_w_pre = Input(shape=(self.T, self.N, 512), dtype='float32', name='WordPreviousHidden') # (B, T, N, 512)
        h_s = Input(shape=(self.T, 1024), dtype='float32', name='SentenceHidden') # (B, T, 1024)

        out = Lambda(lambda x: tf.reshape(tf.tile(x, [1,1,self.N]), [tf.shape(x)[0], self.T, self.N, 1024]))(h_s) # duplicate 1024-embedding for N times (B, T, N, 1024)
        out = concatenate([out, h_w_pre]) # (B, T, N, 1024 + 512)
        out = TimeDistributed(
            Dense(512),
            name='ShrinkDim')(out) # (B, T, N, 512)
        h_w = TimeDistributed(
            LSTM(512, return_sequences=True),
            name="WordRNN")(out) # (B, T, N, 512)
        self.model_2 = Model(inputs=[h_w_pre, h_s], outputs=h_w, name='model_2')

        # Model 3 (3rd part)
        h_w = Input(shape=(self.T, self.N, 512), dtype='float32', name="WordHidden") # (B, T, N, 512)
        out = TimeDistributed(
            Dense(self.V, activation='softmax'),
            name='VocabDistribution')(h_w) # (B, T, N, V)
        self.model_3 = Model(inputs=h_w, outputs=out, name='model_3')

        out = self.model_1([data, h_s_pre])
        out = self.model_2([h_w_pre, out])
        out = self.model_3(out)
        self.model = Model(inputs=[data, h_s_pre, h_w_pre], outputs=out)

class Generator():
    'Create Generator, which generate a next word.'
    def __init__(self, sess, B, V, E, H, lr=1e-3):
        '''
        # Arguments:
            B: int, Batch size
            V: int, Vocabrary size
            E: int, Embedding size
            H: int, LSTM hidden size
        # Optional Arguments:
            lr: float, learning rate, default is 0.001
        '''
        self.sess = sess
        self.B = B
        self.N = 30
        self.V = V
        self.E = E
        self.H = H
        self.lr = lr
        self.layers = []
        self._build_sentence_graph()
        self._build_word_gragh()
        self.reset_rnn_state()

    def _build_sentence_graph(self):
        state_sentence = tf.placeholder(tf.float32, shape=(None, 1, self.N)) # previous generated sentence
        s_h_in = tf.placeholder(tf.float32, shape=(None, 1024)) # sentence RNN initial hidden state
        s_c_in = tf.placeholder(tf.float32, shape=(None, 1024)) # sentence RNN initial cell state
        p_h_in = tf.placeholder(tf.float32, shape=(None, 512)) # paragraph RNN initial hidden state
        p_c_in = tf.placeholder(tf.float32, shape=(None, 512)) # paragraph RNN initial cell state

        wordEmbedding = Embedding(self.V, 512, mask_zero=True, name='WordEmbedding')
        out = wordEmbedding(state_sentence) # (B, 1, N, 512)
        self.layers.append(wordEmbedding)
        
        out = Lambda(lambda x: tf.reduce_mean(x, axis=2), name='SentenceEmbedding')(out) # average word embeddings (B, 1, 512)
        
        paragraphRNN = LSTM(512, return_state=True, name='ParagraphRNN')
        h_p, next_p_h, next_p_c = paragraphRNN(out, initial_state=[p_h_in, p_c_in]) # (B, 512)
        self.layers.append(paragraphRNN)

        out = concatenate([h_p, s_h_in]) # (B, 512 + 1024)
        out = Reshape((1, 512+1024))(out) # (B, 1, 512 + 1024)
        
        expandDense = Dense(4096, name='ExpandDim')
        out = expandDense(out) # (B, 1, 4096)
        self.layers.append(expandDense)

        # TODO: Attention

        sentenceRNN = LSTM(1024, return_state=True, name='SentenceRNN')
        h_s, next_s_h, next_s_c = sentenceRNN(out, initial_state=[s_h_in, s_c_in]) # (B, 1, 1024)
        self.layers.append(sentenceRNN)

        self.h_s_output = h_s # duplicated with word_graph
        self.p_h_in = p_h_in
        self.p_c_in = p_c_in
        self.s_h_in = s_h_in
        self.s_c_in = s_c_in
        self.next_p_h = next_p_h
        self.next_p_c = next_p_c
        self.next_s_h = next_s_h
        self.next_s_c = next_s_c

        self.init_s_op = tf.global_variables_initializer()
        self.sess.run(self.init_s_op)

    def _build_word_gragh(self):
        h_s = tf.placeholder(tf.float32, shape=(None, 1024)) # sentence hidden state
        w_h_in = tf.placeholder(tf.float32, shape=(None, 512)) # word RNN initial hidden state
        w_c_in = tf.placeholder(tf.float32, shape=(None, 512)) # word RNN initial cell state
        action = tf.placeholder(tf.float32, shape=(None, self.V)) # onehot (B, V) next word to select
        reward = tf.placeholder(tf.float32, shape=(None, ))

        out = concatenate([h_s, w_h_in]) # (B, 1024 + 512)
        out = Reshape((1, 1024+512))(out) # (B, 1, 1024 + 512)

        shrinkDense = Dense(512, name='ShrinkDim')
        out = shrinkDense(out)   # (B, 1, 512)
        self.layers.append(shrinkDense)

        # TODO: Attention

        wordRNN = LSTM(512, return_state=True, name='WordRRNN')
        out, next_w_h, next_w_c = wordRNN(out, initial_state=[w_h_in, w_c_in])  # (B, 512)
        self.layers.append(wordRNN)

        vocabDistrib = Dense(self.V, activation='softmax', name='VocabDistribution')
        prob = vocabDistrib(out)   # (B, V)
        self.layers.append(vocabDistrib)

        log_prob = tf.log(tf.reduce_mean(prob * action, axis=-1)) # (B, )
        loss = - log_prob * reward
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        minimize = optimizer.minimize(loss)

        self.h_s_input = h_s
        self.w_h_in = w_h_in
        self.w_c_in = w_c_in
        self.action = action
        self.reward = reward
        self.prob = prob
        self.next_w_h = next_w_h
        self.next_w_c = next_w_c
        self.minimize = minimize
        self.loss = loss

        self.init_w_op = tf.global_variables_initializer()
        self.sess.run(self.init_w_op)

    def reset_rnn_state(self):
        self.h = np.zeros([self.B, self.H])
        self.c = np.zeros([self.B, self.H])

    def set_rnn_state(self, h, c):
        '''
        # Arguments:
            h: np.array, shape = (B,H)
            c: np.array, shape = (B,H)
        '''
        self.h = h
        self.c = c

    def get_rnn_state(self):
        return self.h, self.c

    def predict(self, state, stateful=True):
        '''
        Predict next action(word) probability
        # Arguments:
            state: np.array, previous word ids, shape = (B, 1)
        # Optional Arguments:
            stateful: bool, default is True
                if True, update rnn_state(h, c) to Generator.h, Generator.c
                    and return prob.
                else, return prob, next_h, next_c without updating states.
        # Returns:
            prob: np.array, shape=(B, V)
        '''
        # state = state.reshape(-1, 1)
        feed_dict = {
            self.state_in : state,
            self.h_in : self.h,
            self.c_in : self.c}
        prob, next_h, next_c = self.sess.run(
            [self.prob, self.next_h, self.next_c],
            feed_dict)

        if stateful:
            self.h = next_h
            self.c = next_c
            return prob
        else:
            return prob, next_h, next_c

    def update(self, state, action, reward, h=None, c=None, stateful=True):
        '''
        Update weights by Policy Gradient.
        # Arguments:
            state: np.array, Environment state, shape = (B, 1) or (B, t)
                if shape is (B, t), state[:, -1] will be used.
            action: np.array, Agent action, shape = (B, )
                In training, action will be converted to onehot vector.
                (Onehot shape will be (B, V))
            reward: np.array, reward by Environment, shape = (B, )

        # Optional Arguments:
            h: np.array, shape = (B, H), default is None.
                if None, h will be Generator.h
            c: np.array, shape = (B, H), default is None.
                if None, c will be Generator.c
            stateful: bool, default is True
                if True, update rnn_state(h, c) to Generator.h, Generator.c
                    and return loss.
                else, return loss, next_h, next_c without updating states.

        # Returns:
            loss: np.array, shape = (B, )
            next_h: (if stateful is True)
            next_c: (if stateful is True)
        '''
        if h is None:
            h = self.h
        if c is None:
            c = self.c
        state = state[:, -1].reshape(-1, 1)
        reward = reward.reshape(-1) # (B, 1) -> (B, ) ex. [0.1, 0.9, 0.5...]
        feed_dict = {
            self.state_in : state,
            self.h_in : h,
            self.c_in : c,
            self.action : to_categorical(action, self.V),
            self.reward : reward}
        _, loss, next_h, next_c = self.sess.run(
            [self.minimize, self.loss, self.next_h, self.next_c],
            feed_dict)

        if stateful:
            self.h = next_h
            self.c = next_c
            return loss
        else:
            return loss, next_h, next_c

    def sampling_word(self, prob):
        '''
        # Arguments:
            prob: numpy array, dtype=float, shape = (B, V),
        # Returns:
            action: numpy array, dtype=int, shape = (B, )
        '''
        action = np.zeros((self.B,), dtype=np.int32) # ex. [0, 0, 0, 0, 0 ...]
        for i in range(self.B):
            p = prob[i]
            action[i] = np.random.choice(self.V, p=p) # random select a int value in range(self.V) according to the prob distrbution p
        return action # (B, ) word ids

    def sampling_sentence(self, T, BOS=1):
        '''
        # Arguments:
            T: int, max time steps
        # Optional Arguments:
            BOS: int, id for Begin Of Sentence
        # Returns:
            actions: numpy array, dtype=int, shape = (B, T)
        '''
        self.reset_rnn_state() # set Generator LSTM h, c state to zero vectors
        action = np.zeros([self.B, 1], dtype=np.int32)
        action[:, 0] = BOS # ex. [[1], [1], [1]...]
        actions = action
        for _ in range(T):
            prob = self.predict(action) # (B, V)
            action = self.sampling_word(prob).reshape(-1, 1) # (B, 1) ex. [[20], [2239], [word id]...]
            actions = np.concatenate([actions, action], axis=-1)  # (B, T) ex. [[1, 20], [1, 2239], [BOS, word id]...]
        # Remove BOS
        actions = actions[:, 1:]
        self.reset_rnn_state()
        return actions

    def generate_samples(self, T, g_data, num, output_file):
        '''
        Generate sample sentences to output file
        # Arguments:
            T: int, max time steps
            g_data: SeqGAN.utils.GeneratorPretrainingGenerator
            num: int, number of sentences
            output_file: str, path
        '''
        sentences=[]
        for _ in range(num // self.B + 1):
            actions = self.sampling_sentence(T) # (B, T)
            actions_list = actions.tolist()
            for sentence_id in actions_list:
                sentence = [g_data.id2word[action] for action in sentence_id]
                sentences.append(sentence)
        output_str = ''
        for i in range(num):
            output_str += ' '.join(sentences[i]) + '\n'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_str)

    def save(self, path):
        weights = []
        for layer in self.layers:
            w = layer.get_weights()
            weights.append(w)
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)

def Discriminator(V, E, H=64, dropout=0.1):
    '''
    Disciriminator model.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        H: int, LSTM hidden size
        dropout: float
    # Returns:
        discriminator: keras model
            input: word ids, shape = (B, T)
            output: probability of true data or not, shape = (B, 1)
    '''
    input = Input(shape=(None,), dtype='int32', name='Input')   # (B, T)
    out = Embedding(V, E, mask_zero=True, name='Embedding')(input)  # (B, T, E)
    out = LSTM(H)(out) # (B, H)
    out = Highway(out, num_layers=1)
    out = Dropout(dropout, name='Dropout')(out)
    out = Dense(1, activation='sigmoid', name='FC')(out) # (B, 1)

    discriminator = Model(input, out)
    return discriminator

def DiscriminatorConv(V, E, filter_sizes, num_filters, dropout):
    '''
    Another Discriminator model, currently unused because keras don't support
    masking for Conv1D and it does huge influence on training.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        filter_sizes: list of int, list of each Conv1D filter sizes
        num_filters: list of int, list of each Conv1D num of filters
        dropout: float
    # Returns:
        discriminator: keras model
            input: word ids, shape = (B, T)
            output: probability of true data or not, shape = (B, 1)
    '''
    input = Input(shape=(None,), dtype='int32', name='Input')   # (B, T)
    out = Embedding(V, E, name='Embedding')(input)  # (B, T, E)
    out = VariousConv1D(out, filter_sizes, num_filters)
    out = Highway(out, num_layers=1)
    out = Dropout(dropout, name='Dropout')(out)
    out = Dense(1, activation='sigmoid', name='FC')(out)

    discriminator = Model(input, out)
    return discriminator

def VariousConv1D(x, filter_sizes, num_filters, name_prefix=''):
    '''
    Layer wrapper function for various filter sizes Conv1Ds
    # Arguments:
        x: tensor, shape = (B, T, E)
        filter_sizes: list of int, list of each Conv1D filter sizes
        num_filters: list of int, list of each Conv1D num of filters
        name_prefix: str, layer name prefix
    # Returns:
        out: tensor, shape = (B, sum(num_filters))
    '''
    conv_outputs = []
    for filter_size, n_filter in zip(filter_sizes, num_filters):
        conv_name = '{}VariousConv1D/Conv1D/filter_size_{}'.format(name_prefix, filter_size)
        pooling_name = '{}VariousConv1D/MaxPooling/filter_size_{}'.format(name_prefix, filter_size)
        conv_out = Conv1D(n_filter, filter_size, name=conv_name)(x)   # (B, time_steps, n_filter)
        conv_out = GlobalMaxPooling1D(name=pooling_name)(conv_out) # (B, n_filter)
        conv_outputs.append(conv_out)
    concatenate_name = '{}VariousConv1D/Concatenate'.format(name_prefix)
    out = Concatenate(name=concatenate_name)(conv_outputs)
    return out

def Highway(x, num_layers=1, activation='relu', name_prefix=''):
    '''
    Layer wrapper function for Highway network
    # Arguments:
        x: tensor, shape = (B, input_size)
    # Optional Arguments:
        num_layers: int, dafault is 1, the number of Highway network layers
        activation: keras activation, default is 'relu'
        name_prefix: str, default is '', layer name prefix
    # Returns:
        out: tensor, shape = (B, input_size)
    '''
    input_size = K.int_shape(x)[1]
    for i in range(num_layers):
        gate_ratio_name = '{}Highway/Gate_ratio_{}'.format(name_prefix, i)
        fc_name = '{}Highway/FC_{}'.format(name_prefix, i)
        gate_name = '{}Highway/Gate_{}'.format(name_prefix, i)

        gate_ratio = Dense(input_size, activation='sigmoid', name=gate_ratio_name)(x)
        fc = Dense(input_size, activation=activation, name=fc_name)(x)
        x = Lambda(lambda args: args[0] * args[2] + args[1] * (1 - args[2]), name=gate_name)([fc, x, gate_ratio])
    return x
