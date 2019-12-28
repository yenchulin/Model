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
from tqdm import trange

def visualAttention(x):
    '''
    # Arguments:
        x: hidden state from LSTM, shape (B, 4096), a tensor
    # Variables:
        feature: tensor with shape (1, 50, 4096), 50 visual features with 4096 dim
    # Returns:
        context: tensor with shape (B, 4096), visual context vector
    '''
    feature = np.full([50, 4096], 1) # TODO: should be replaced
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
        
        # TODO: Attention

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

    def train_on_batch(self, data, epochs):
        '''
        # Arguments:
            data: Seqgan.utils.DiscriminatorGenerator
            epochs: int, number of epochs to run
        # Returns:
            losses: nparray, loss of every epoch, shape = (epochs, )
        # Note: Have to compile the model first.
        '''
        losses = []
        for epoch in range(epochs):
            with trange(len(data), ascii=True) as num_batch: # Total number of steps (number of batches = num_samples / batch_size)
                num_batch.set_description("Epoch %i/%i" % (epoch+1, epochs))
                loss = 0
                for _ in num_batch: 
                    sample = data.next()
                    loss += self.model.train_on_batch(
                        x=sample[0],
                        y=sample[1],
                        sample_weight=sample[2]
                    ) / len(data) # average the loss in same batch
                    num_batch.set_postfix(loss=loss)
            
            losses.append(loss)
        losses = np.array(losses, dtype=np.float)
        return losses

class Generator():
    'Create Generator, which generate a next word.'
    def __init__(self, sess, B, N, V, E, H, lr=1e-3):
        '''
        # Arguments:
            B: int, Batch size
            N: int, Max words in a sentence
            V: int, Vocabrary size
            E: int, Embedding size
            H: int, LSTM hidden size
        # Optional Arguments:
            lr: float, learning rate, default is 0.001
        '''
        self.sess = sess
        self.B = B
        self.N = N
        self.V = V
        self.E = E
        self.H = H
        self.lr = lr
        self.layers = []
        self._build_sentence_graph()
        self._build_word_gragh()
        self.reset_rnn_state()

    def _build_sentence_graph(self):
        state_sentence = tf.placeholder(tf.float32, shape=(None, self.N)) # previous generated sentence
        s_h_in = tf.placeholder(tf.float32, shape=(None, 1024)) # sentence RNN initial hidden state
        s_c_in = tf.placeholder(tf.float32, shape=(None, 1024)) # sentence RNN initial cell state
        p_h_in = tf.placeholder(tf.float32, shape=(None, 512)) # paragraph RNN initial hidden state
        p_c_in = tf.placeholder(tf.float32, shape=(None, 512)) # paragraph RNN initial cell state

        out = Reshape((1, self.N))(state_sentence) # (B, 1, N)
        
        wordEmbedding = Embedding(self.V, 512, mask_zero=True, name='WordEmbedding')
        out = wordEmbedding(out) # (B, 1, N, 512)
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
        h_s, next_s_h, next_s_c = sentenceRNN(out, initial_state=[s_h_in, s_c_in]) # (B, 1024)
        self.layers.append(sentenceRNN)

        self.state_sentence = state_sentence
        self.p_h_in = p_h_in
        self.p_c_in = p_c_in
        self.s_h_in = s_h_in
        self.s_c_in = s_c_in
        self.h_s_output = h_s
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
        self.p_h = np.zeros([self.B, 512])
        self.p_c = np.zeros([self.B, 512])
        self.s_h = np.zeros([self.B, 1024])
        self.s_c = np.zeros([self.B, 1024])
        self.w_h = np.zeros([self.B, 512])
        self.w_c = np.zeros([self.B, 512])

    def set_rnn_state(self, p_h, p_c, s_h, s_c, w_h, w_c):
        '''
        # Arguments:
            h: np.array, shape = (B,H)
            c: np.array, shape = (B,H)
        '''
        self.p_h = p_h
        self.p_c = p_c
        self.s_h = s_h
        self.s_c = s_c
        self.w_h = w_h
        self.w_c = w_c

    def get_rnn_state(self):
        return self.p_h, self.p_c, self.s_h, self.s_c, self.w_h, self.w_c

    def predict_h_s(self, state_sentence):
        """
        # Arguments:
            state_sentence: nparray, dtype = int, shape = (B, N)
        """
        feed_dict = {
            self.state_sentence: state_sentence,
            self.p_h_in: self.p_h,
            self.p_c_in: self.p_c,
            self.s_h_in: self.s_h,
            self.s_c_in: self.s_c
        }
        h_s, next_s_h, next_s_c, next_p_h, next_p_c = self.sess.run(
            [self.h_s_output, self.next_s_h, self.next_s_c, self.next_p_h, self.next_p_c],
            feed_dict)
        
        self.s_h = next_s_h
        self.s_c = next_s_c
        self.p_h = next_p_h
        self.p_c = next_p_c
        return h_s

    def predict_word(self, h_s):
        feed_dict = {
            self.h_s_input: h_s,
            self.w_h_in: self.w_h,
            self.w_c_in: self.w_c
        }
        prob, next_w_h, next_w_c = self.sess.run(
            [self.prob, self.next_w_h, self.next_w_c],
            feed_dict)
        
        self.w_h = next_w_h
        self.w_c = next_w_c
        return prob

    def update(self, state, action, reward):
        '''
        Update weights by Policy Gradient.
        # Arguments:
            state: nparray, Environment state (previous sentence), shape = (B, N)
            action: nparray, Agent action (word), shape = (B, 1)
                In training, action will be converted to onehot vector.
                (Onehot shape will be (B, V))
            reward: nparray, reward by Environment, shape = (B, 1)

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
            loss: np.array, shape = (B, ) -> float, average the loss of every sample in a batch
            next_h: (if stateful is True)
            next_c: (if stateful is True)
        '''
        feed_dict = {
            self.h_s_input: self.predict_h_s(state),
            self.w_h_in : self.w_h,
            self.w_c_in : self.w_c,
            self.action : to_categorical(action, self.V),
            self.reward : reward.reshape(-1)
        }
        _, loss, next_w_h, next_w_c = self.sess.run(
            [self.minimize, self.loss, self.next_w_h, self.next_w_h],
            feed_dict)

        self.w_h = next_w_h
        self.w_c = next_w_c
        return np.mean(loss) # a float value

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

    def sampling_paragraph(self, T):
        self.reset_rnn_state()
        paragraph = []

        # previous sentence initial value 
        sentence = np.zeros([self.B, self.N], dtype=np.int32)
        sentence[:, 0] = 1 # BOS
        sentence[:, 1] = 2 # EOS
        
        for _ in range(T):
            h_s = self.predict_h_s(sentence)
            sentence = np.zeros([self.B, 0], dtype=np.int32)
            for _ in range(self.N):
                prob = self.predict_word(h_s)
                action = self.sampling_word(prob).reshape(-1, 1)
                sentence = np.concatenate([sentence, action], axis=-1) # (B, N)
            paragraph.append(sentence) # (T, B, N)

        paragraph = np.array(paragraph)
        paragraph = paragraph.transpose([1, 0, 2]) # (B, T, N)
        self.reset_rnn_state()
        return paragraph

    def generate_samples(self, T, g_data, num, output_file):
        '''
        Generate sample paragraphs to output file
        # Arguments:
            T: int, Max sentences in a paragraph
            g_data: SeqGAN.utils.GeneratorPretrainingGenerator
            num: int, number of sample paragraphs to generate
            output_file: str, path
        '''
        paragraphs = []
        for i in range(num // self.B + 1):
            batch_paragraphs = self.sampling_paragraph(T) # paragraph with word ids (B, T, N)
            batch_paragraphs = batch_paragraphs.tolist()
            for paragraph in batch_paragraphs:
                para = []
                for sentence in paragraph:
                    sentence = [g_data.vocab.id2word[word] for word in sentence]
                    sentence = " ".join(sentence)
                    para.append(sentence)
                paragraphs.append(para)

        output_str = ''
        for i in range(num):
            output_str += ' '.join(paragraphs[i]) + '\n'
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

def DisciriminatorParagraph(B, T, N, V, dropout=0.1):
    '''
    Paragraph Disciriminator model.
    # Arguments:
        B: int, Batch size
        T: int, Max sentences in a paragraph
        N: int, Max words in a sentence
        V: int, Vocabrary size
        dropout: float
    # Returns:
        discriminator: keras model
            input: sentences, shape = (B, T, N)
            output: probability (smoothness vlllue) of true sentence or not, shape = (B, T, 1)
    '''
    input = Input(shape=(T, N), dtype='int32', name='Input')  # (B, T, N)
    out = Embedding(V, 512, mask_zero=True, name='WordEmbedding')(input)  # (B, T, N, 512)
    out = Lambda(lambda x: tf.reduce_mean(x, axis=2), name="SentenceEmbedding")(out) # average word embeddings (B, T, 512)
    out = LSTM(512, return_sequences=True)(out) # (B, T, 512)
    out = Highway(out, num_layers=1) # (B, T, 512)
    out = Dropout(dropout, name='Dropout')(out) # (B, T, 512)
    out = Dense(1, activation='sigmoid', name='FC')(out) # (B, T, 1)

    discriminator = Model(input, out)
    return discriminator

class DiscriminatorSentence():
    '''
    Sentence Disciriminator model.
    # Arguments:
        V: int, Vocabrary size
        dropout: float
    # Parameters:
        model: the sentence discriminator model
    '''
    def __init__(self, V, dropout=0.1):
        self.__build_graph__(V, dropout)
    
    def __build_graph__(self, V, dropout):
        input = Input(shape=(None,), dtype='int32', name='Input')  # (B, N)
        out = Embedding(V, 512, mask_zero=True, name='WordEmbedding')(input)  # (B, N, 512)
        out = LSTM(512)(out) # (B, 512)
        out = Highway(out, num_layers=1) # (B, 512)
        out = Dropout(dropout, name='Dropout')(out) # (B, 512)
        out = Dense(1, activation='sigmoid', name='FC')(out) # (B, 1)

        self.model = Model(input, out)

    def train_on_batch(self, data, epochs):
        '''
        # Arguments:
            data: Seqgan.utils.DiscriminatorGenerator
            epochs: int, number of epochs to run
        # Returns:
            losses: nparray, loss of every epoch, shape = (epochs, )
        # Note: Have to compile the model first.
        '''
        losses = []
        for epoch in range(epochs):
            with trange(len(data), ascii=True) as num_batch: # Total number of steps (number of batches = num_samples / batch_size)
                num_batch.set_description("Epoch %i/%i" % (epoch+1, epochs))
                loss = 0
                for _ in num_batch:
                    sample = data.next()
                    loss += self.model.train_on_batch(
                        x=sample[0],
                        y=sample[1]
                    ) / len(data) # average the loss in same batch
                    num_batch.set_postfix(loss=loss)

            losses.append(loss)
        losses = np.array(losses, dtype=np.float)
        return losses

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
    input_size = K.int_shape(x)[-1]
    for i in range(num_layers):
        gate_ratio_name = '{}Highway/Gate_ratio_{}'.format(name_prefix, i)
        fc_name = '{}Highway/FC_{}'.format(name_prefix, i)
        gate_name = '{}Highway/Gate_{}'.format(name_prefix, i)

        gate_ratio = Dense(input_size, activation='sigmoid', name=gate_ratio_name)(x)
        fc = Dense(input_size, activation=activation, name=fc_name)(x)
        x = Lambda(lambda args: args[0] * args[2] + args[1] * (1 - args[2]), name=gate_name)([fc, x, gate_ratio])
    return x
