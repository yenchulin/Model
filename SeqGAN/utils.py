import numpy as np
import random
import linecache
import re
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical

def plotLineChart(x, y, xlabelName, ylabelName, figname):
    plt.figure(figsize=(13,7))
    
    # create the line plot
    plt.plot(x, y)
    plt.xticks(x)
    plt.xlabel(xlabelName)
    plt.ylabel(ylabelName)
    
    # save the plot
    plt.savefig(figname)

class Vocab:
    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<S>'
    EOS_TOKEN = '</S>'

    def __init__(self, file_path, min_count=1):
        """
        # Arguments:
            file_path: str
            min_count: int, min count of a word to add into Vocab.
        # Parameters:
            word2id: a dictionary with word as key, and word_id as value.
            id2word: a dictionary with word_id as key, and word as value.
            raw_vocab: a dictionary with word as key, word count as value.
            V: int, the size of the vocabulary
        """
        default_dict = {
            Vocab.PAD_TOKEN: Vocab.PAD,
            Vocab.BOS_TOKEN: Vocab.BOS,
            Vocab.EOS_TOKEN: Vocab.EOS,
            Vocab.UNK_TOKEN: Vocab.UNK
        }
        self.word2id = dict(default_dict)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.raw_vocab = {}

        _, data = load_data(file_path)
        self.__build_vocab__(data, min_count)

    def __build_vocab__(self, data, min_count=1):
        """
        Build vocabulary for given data.
        # Arguments:
            data: list, shape = (n_data, number of sentences in each row, number of words in each sentence).
            min_count: int, min count of a word to add into Vocab.
        """
        word_counter = {}
        for paragraph in data:
            for sentence in paragraph:
                for word in sentence:
                    word_counter[word] = word_counter.get(word, 0) + 1
        
        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
            if count < min_count:
                continue
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word
            self.raw_vocab[word] = count
        self.V = len(self.word2id)

def load_sentence_data(file_path):
    """
    Load string data that each line represents a sample (sentence).
    Every sample has puntuations.
    Length don't have to be fixed.
    # Arguments:
        file_path: str
    # Returns:
        n_data: int, number of rows (sample) in file
        data: list, shape = (n_data, number of words in each sentence)
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = re.sub("[^a-zA-Z0-9]", " ", line)
            words = sentence.strip().split() # list of str
            data.append(words)
    
    return len(data), data

def load_data(file_path):
    """
    Load string data that each line represents a sample (paragraph).
    Every sample has puntuations to seperate into sentences.
    Length don't have to be fixed.
    # Arguments:
        file_path: str
    # Returns:
        n_data: int, number of rows (sample) in file
        data: list, shape = (n_data, number of sentences in each row, number of words in each sentence)
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            paragraph = []
            for sentence in sent_tokenize(line):
                sentence = re.sub("[^a-zA-Z0-9]", " ", sentence)
                words = sentence.strip().split() # list of str
                paragraph.append(words)
            data.append(paragraph)
    
    return len(data), data

def load_generated_data(file_path, T, N):
    """
    Load Generator generated data that each line represents a sample (paragraph).
    Every sample has T * N tokens.
    # Arguments:
        file_path: str
        T: int, number of sentence in a sample (paragraph)
        N: int, number of words in a sentence
    # Returns:
        n_data: int, number of rows (sample) in file
        data: list, shape = (n_data, number of sentences in each row, number of words in each sentence)
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            paragraph = line.strip().split()
            if len(paragraph) != T * N:
                raise ValueError("expected sample length %d but get %d" % (T * N, len(paragraph)))
            paragraph = [paragraph[x:x+N] for x in range(0, len(paragraph), N)] # (T, N)
            data.append(paragraph)
    
    return len(data), data

def get_sentence_ids(data_row, vocab):
    '''
    # Arguments:
        data_row: a sample from data, list, shape = (number of words in each sentence, )
        vocab: SeqGAN.utils.Vocab used for lookup word ids
    # Returns:
        sentence: list of int
    '''
    return [vocab.word2id.get(word, Vocab.UNK) for word in data_row]

def get_paragraph_ids(data_row, vocab):
    '''
    # Arguments:
        data_row: a sample from data, list, shape = (number of sentences in each row, number of words in each sentence)
        vocab: SeqGAN.utils.Vocab used for lookup word ids
    # Returns:
        paragraph: list of int, 
                data[i] means a sentence, (i=1)
                data[i][j] means a word.
    '''
    paragraph = []
    for sentence in data_row:
        ids = [vocab.word2id.get(word, Vocab.UNK) for word in sentence]
        paragraph.append(ids) # ex. [[BOS(1), 8, 10, 6, 3, EOS(2)], [1, 5, 13, 9, 7, 25, 2]]
    return paragraph

def reshape_sentence(data_row, max_num_words, BOS=False, EOS=False):
    """
    # Arguments:
        data_row: a sample from data, list, shape = (number of words in each sentence, )
        max_num_words: int.
        BOS (optional): the BOS token, default is False. If False, BOS will not be added at the begin of the sentence.
        EOS (optional): the EOS token, default is False. If False, EOS will not be added at the end of the sentence.
    # Returns:
        sentence: a sentence, list, shape = (max_num_words, ).
    """
    sentence = list(data_row)
    sentence_len = len(sentence)
    if BOS:
        sentence_len += 1
    if EOS:
        sentence_len += 1
    
    # Check if need pad or drop
    diff = max_num_words - sentence_len
    if diff > 0: # pad
        if BOS:
            sentence.insert(0, Vocab.BOS_TOKEN)
        if EOS:
            sentence.append(Vocab.EOS_TOKEN)
        sentence.extend([Vocab.PAD_TOKEN] * diff)
    elif diff < 0: # drop
        sentence = sentence[:diff]
        if BOS:
            sentence.insert(0, Vocab.BOS_TOKEN)
        if EOS:
            sentence.append(Vocab.EOS_TOKEN)
    else: # no need to drop or pad
        if BOS:
            sentence.insert(0, Vocab.BOS_TOKEN)
        if EOS:
            sentence.append(Vocab.EOS_TOKEN)
    return sentence

def reshape_paragraph(data_row, max_num_sentences, max_num_words, BOS=False, EOS=False):
    """
    # Arguments:
        data_row: a sample from data, list, shape = (number of sentences in each row, number of words in each sentence)
        max_num_sentences: int.
        max_num_words: int.
        BOS (optional): the BOS token, default is False. If False, BOS will not be added at the begin of the sentence.
        EOS (optional): the EOS token, default is False. If False, EOS will not be added at the end of the sentence.
    # Returns:
        paragraph: a paragraph, list, shape = (max_num_sentences, max_num_words).
    """
    paragraph = []
    for sentence in data_row:
        sentence = list(sentence)
        sentence_len = len(sentence)
        if BOS:
            sentence_len += 1
        if EOS:
            sentence_len += 1
        
        # Check if need pad or drop
        diff = max_num_words - sentence_len
        if diff > 0: # pad
            if BOS:
                sentence.insert(0, Vocab.BOS_TOKEN)
            if EOS:
                sentence.append(Vocab.EOS_TOKEN)
            sentence.extend([Vocab.PAD_TOKEN] * diff)
        elif diff < 0: # drop
            sentence = sentence[:diff]
            if BOS:
                sentence.insert(0, Vocab.BOS_TOKEN)
            if EOS:
                sentence.append(Vocab.EOS_TOKEN)
        else: # no need to drop or pad
            if BOS:
                sentence.insert(0, Vocab.BOS_TOKEN)
            if EOS:
                sentence.append(Vocab.EOS_TOKEN)
        paragraph.append(sentence)

    paragraph_len = len(paragraph)
    diff = max_num_sentences - paragraph_len
    if diff > 0: # pad
        sentence_pad = [Vocab.PAD_TOKEN] * max_num_words
        if BOS and EOS:
            sentence_pad[0] = Vocab.BOS_TOKEN
            sentence_pad[1] = Vocab.EOS_TOKEN
        elif BOS and not EOS:
            sentence_pad[0] = Vocab.BOS_TOKEN
        elif not BOS and EOS:
            sentence_pad[0] = Vocab.EOS_TOKEN
        paragraph.extend([sentence_pad] * diff)
    elif diff < 0: # drop
        paragraph = paragraph[:diff]
    
    return paragraph

def print_ids(ids, vocab, verbose=True, exclude_mark=True, PAD=0, BOS=1, EOS=2):
    '''
    :param ids: list of int,
    :param vocab:
    :param verbose(optional): 
    :return sentence: list of str
    '''
    sentence = []
    for i, id in enumerate(ids):
        word = vocab.id2word[id]
        if exclude_mark and id == EOS:
            break
        if exclude_mark and id in (BOS, PAD):
            continue
        sentence.append(sentence)
    if verbose:
        print(sentence)
    return sentence


class GeneratorPretrainingGenerator(Sequence):
    '''
    Generate generator pretraining data.
    # Arguments
        path: str, path to data x
        B: int, batch size
        T: the max number of sentences in a document (review)
        N: the max number of words in a sentence
        vocab: Vocab
        shuffle (optional): bool

    # Params
        n_data: the number of rows of data
        data: list, shape = (n_data, number of sentences in each row, number of words in each sentence)
        graph: tf.get_default_graph(), force tf to run model in the same session
        model_s: keras.layer.Model, used to get h_s_pre
        model_w: keras.layer.Model, used to get h_w_pre
    '''
    def __init__(self, path, B, T, N, vocab, shuffle=True):
        self.B = B
        self.T = T
        self.N = N
       
        self.vocab = vocab
        self.n_data, self.data = load_data(path)
        self.shuffle = shuffle
        self.idx = 0
        self.len = self.__len__()
        self.reset()
        self.graph = None
        self.model_s = None
        self.model_w = None
        self.h_s_pre_pre = np.random.rand(self.B, self.T, 1024) # sentence hidden state at B-2 (B, T ,1024)
        self.h_w_pre_pre = np.random.rand(self.B, self.T, self.N, 512) # word hidden state at B-2 (B, T, N, 512)
        self.x_pre = None # data at B-1 (B, T, N)


    def __len__(self):
        return self.n_data // self.B

    def __getitem__(self, idx):
        '''
        Get generator pretraining data batch.
        # Arguments:
            idx: int, index of batch
        # Returns:
            x: numpy.array, shape = (B, T, N)
            y_true: numpy.array, shape = (B, T, N, V)
                labels with one-hot encoding.
                
                T is the max number of sentences in the batch.
                if number of sentences is smaller than max_num_sentences, the data will be padded.

                N is the max number of words in the batch.
                if number of sentences is smaller than max_num_words, the data will be padded.
        '''
        x, y_true = [], []
        start = idx * self.B
        end = (idx + 1) * self.B

        for i in range(start, end):
            if self.shuffle:
                idx = self.shuffled_indices[i]
            else:
                idx = i

            paragraph = self.data[idx] # a sample from data, list, shape = (number of sentences in each row, number of words in each sentence)
            
            paragraph_x = reshape_paragraph(paragraph, self.T, self.N, BOS=True, EOS=True)
            paragraph_ids_x = get_paragraph_ids(paragraph_x, self.vocab)
            x.append(paragraph_ids_x)

            paragraph_y_true = reshape_paragraph(paragraph, self.T, self.N, EOS=True)
            paragraph_ids_y_true = get_paragraph_ids(paragraph_y_true, self.vocab)
            y_true.append(paragraph_ids_y_true)

        x = np.array(x, dtype=np.int32)

        y_true = np.array(y_true, dtype=np.int32)
        y_true = to_categorical(y_true, num_classes=self.vocab.V) # (B, T, N, V)
 
        if self.model_s != None and self.model_w != None:
            self.x_pre = x
            if idx == 0:
                return ([x, self.h_s_pre_pre, self.h_w_pre_pre], y_true)
            else:
                with self.graph.as_default():
                    h_s_pre = self.model_s.predict([self.x_pre, self.h_s_pre_pre]) # sentence hidden state at B-1 (B, T ,1024)
                    h_w_pre = self.model_w.predict([self.h_w_pre_pre, h_s_pre]) # word hidden state at B-1 (B, T, N, 512)
                self.h_s_pre_pre = h_s_pre
                self.h_w_pre_pre = h_w_pre
                return ([x, h_s_pre, h_w_pre], y_true)
        else:
            return ([x], y_true)

    def __iter__(self):
        return self

    def next(self):
        if self.idx >= self.len:
            self.reset()

        x, y_true = self.__getitem__(self.idx)
        loss_weight = self.sample_loss_weight(x)

        self.idx += 1
        return (x, y_true, loss_weight)

    def sample_loss_weight(self, x):
        """
        Every sentence's weight (score) will be 30. (will be normalized in keras, no need to manage)
        So if a sentence has one <PAD> (id=0), the score will minus 1.
        """
        sample = x[0] # (B, T, N)
        loss_weight = np.count_nonzero(sample, axis=-1) # (B, T)
        return loss_weight

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.shuffled_indices = np.arange(self.n_data)
            random.shuffle(self.shuffled_indices)

    def on_epoch_end(self):
        self.reset()
        pass


class DiscriminatorSentenceGenerator(Sequence):
    '''
    Generate Sentence Discriminator data.
    # Arguments
        path_pos: str, path to true data
        path_neg: str, path to generated data
        B: int, batch size
        T: the max number of sentences in a document (review)
        N: the max number of words in a sentence
        vocab: Vocab
        shuffle (optional): bool

    # Params
        n_data: the number of rows of data
    '''
    def __init__(self, path_pos, path_neg, B, T, N, vocab, shuffle=True):
        self.B = B
        self.T = T
        self.N = N
        
        self.vocab = vocab
        self.n_data_pos, self.data_pos = load_sentence_data(path_pos)
        self.n_data_neg, self.data_neg = load_generated_data(path_neg, self.T, self.N)
        self.n_data = self.n_data_pos + self.n_data_neg
        self.shuffle = shuffle
        self.idx = 0
        self.len = self.__len__()
        self.reset()

    def __len__(self):
        return self.n_data // self.B

    def __getitem__(self, idx):
        '''
        Get generator pretraining data batch.
        # Arguments:
            idx: int, index of batch
        # Returns:
            X: numpy.array, shape = (B, N)
            Y: numpy.array, shape = (B, 1)
                labels indicate whether sentences are true data or generated data.
                if true data, y = 1. Else if generated data, y = 0.
        '''
        X, Y = [], []
        start = idx * self.B
        end = (idx + 1) * self.B

        for i in range(start, end):
            idx = self.indicies[i]
            is_pos = 1
            if idx < 0:
                is_pos = 0
                idx = -1 * idx
            idx = idx - 1

            if is_pos == 1:
                sentence = self.data_pos[idx] # a sample from data, list, shape = (number of words in each sentence, )
                sentence_x = reshape_sentence(sentence, self.N, EOS=True) # (N, )
                sentence_ids_x = get_sentence_ids(sentence_x, self.vocab)
                X.append(sentence_ids_x)
                Y.append([is_pos])
            elif is_pos == 0:
                paragraph = self.data_neg[idx] # a sample from data, list, shape = (number of sentences in each row, number of words in each sentence)
                paragraph_ids_x = get_paragraph_ids(paragraph, self.vocab) # (T, N)
                for sentence_ids_x in paragraph_ids_x:
                    X.append(sentence_ids_x)
                    Y.append([is_pos])

        X = np.array(X, dtype=np.int32)
        Y = np.array(Y, dtype=np.int32)
        return (X, Y)

    def __iter__(self):
        return self

    def next(self):
        if self.idx >= self.len:
            self.reset()

        X, Y = self.__getitem__(self.idx)
        self.idx += 1
        return (X, Y)

    def reset(self):
        self.idx = 0
        pos_indices = np.arange(start=1, stop=self.n_data_pos+1)
        neg_indices = -1 * np.arange(start=1, stop=self.n_data_neg+1)
        self.indicies = np.concatenate([pos_indices, neg_indices])
        if self.shuffle:
            random.shuffle(self.indicies)

    def on_epoch_end(self):
        self.reset()
        pass


class DiscriminatorGenerator(Sequence):
    '''
    Generate generator pretraining data.
    # Arguments
        path_pos: str, path to true data
        path_neg: str, path to generated data
        B: int, batch size
        T: the max number of sentences in a document (review)
        N: the max number of words in a sentence
        vocab: Vocab
        shuffle (optional): bool

    # Params
        n_data: the number of rows of data
    '''
    def __init__(self, path_pos, path_neg, B, T, N, vocab, shuffle=True):
        self.B = B
        self.T = T
        self.N = N
        
        self.vocab = vocab
        self.n_data_pos, self.data_pos = load_data(path_pos)
        self.n_data_neg, self.data_neg = load_generated_data(path_neg, self.T, self.N)
        self.n_data = self.n_data_pos + self.n_data_neg
        self.shuffle = shuffle
        self.idx = 0
        self.len = self.__len__()
        self.reset()

    def __len__(self):
        return self.n_data // self.B

    def __getitem__(self, idx):
        '''
        Get generator pretraining data batch.
        # Arguments:
            idx: int, index of batch
        # Returns:
            None: no input is needed for generator pretraining.
            X: numpy.array, shape = (B, T, N)
            Y: numpy.array, shape = (B, T, 1)
                labels indicate whether sentences are true data or generated data.
                if true data, y = 1. Else if generated data, y = 0.
        '''
        X, Y = [], []
        start = idx * self.B
        end = (idx + 1) * self.B

        for i in range(start, end):
            idx = self.indicies[i]
            is_pos = 1
            if idx < 0:
                is_pos = 0
                idx = -1 * idx
            idx = idx - 1

            if is_pos == 1:
                paragraph = self.data_pos[idx] # a sample from data, list, shape = (number of sentences in each row, number of words in each sentence)
                paragraph_x = reshape_paragraph(paragraph, self.T, self.N, EOS=True)
                paragraph_ids_x = get_paragraph_ids(paragraph_x, self.vocab)
            elif is_pos == 0:
                paragraph = self.data_neg[idx] # a sample from data, list, shape = (number of sentences in each row, number of words in each sentence)
                paragraph_ids_x = get_paragraph_ids(paragraph, self.vocab)

            X.append(paragraph_ids_x)
            Y.append([[is_pos]] * self.T)

        X = np.array(X, dtype=np.int32).reshape(self.B*self.T, self.N) # (B*T, N) reshape for DiscriminatorSentence, not tested

        Y = np.array(Y, dtype=np.int32).reshape(self.B*self.T, 1) # (B*T, 1) reshape for DiscriminatorSentence, not tested
        return (X, Y)

    def __iter__(self):
        return self

    def next(self):
        if self.idx >= self.len:
            self.reset()

        X, Y = self.__getitem__(self.idx)
        self.idx += 1
        return (X, Y)

    def reset(self):
        self.idx = 0
        pos_indices = np.arange(start=1, stop=self.n_data_pos+1)
        neg_indices = -1 * np.arange(start=1, stop=self.n_data_neg+1)
        self.indicies = np.concatenate([pos_indices, neg_indices])
        if self.shuffle:
            random.shuffle(self.indicies)

    def on_epoch_end(self):
        self.reset()
        pass
