import numpy as np
import random
import linecache
import re
from nltk.tokenize import sent_tokenize
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical

class Vocab:
    def __init__(self, word2id, unk_token):
        self.word2id = dict(word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.unk_token = unk_token

    def build_vocab(self, data, min_count=1):
        word_counter = {}
        for line in data:
            for word in line:
                word_counter[word] = word_counter.get(word, 0) + 1

        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
            if count < min_count:
                break
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word

        self.raw_vocab = {w: word_counter[w] for w in self.word2id.keys() if w in word_counter}

    def sentence_to_ids(self, sentence):
        return [self.word2id[word] if word in self.word2id else self.word2id[self.unk_token] for word in sentence]

def load_data(file_path):
    '''
    # Arguments:
        file_path: str
    # Returns:
        data: list of list of str, data[i] means a line, data[i][j] means a
             word.
    '''
    data = []
    for line in open(file_path, encoding='utf-8'):
        words = line.strip().split()
        data.append(words)
    return data

def sentence_to_ids(vocab, sentence, UNK=3):
    '''
    # Arguments:
        vocab: SeqGAN.utils.Vocab
        sentence: list of str
    # Returns:
        ids: list of int
    '''
    ids = [vocab.word2id.get(word, UNK) for word in sentence]
    # ids += [EOS]  # EOSを加える
    return ids

def get_paragraph_ids(data_row, vocab, BOS=None, EOS=None):
    '''
    # Arguments:
        data_row: a line from the data
        vocab: SeqGAN.utils.Vocab used for lookup word ids
        BOS (optional): the BOS id, default is None. If is None, BOS will not be added at the begin of the sentence.
        EOS (optional): the EOS id, default is None. If is None, EOS will not be added at the end of the sentence.
    # Returns:
        paragraph: list of list of list of ids, 
                data[i] means a paragraph, (i=1)
                data[i][j] means a sentence,
                data[i][j][k] means a word.
    '''
    paragraph = []
    for sentence in sent_tokenize(data_row):
        sentence = re.sub("[^a-zA-Z0-9]", " ", sentence)
        words = sentence.strip().split() # list of str

        word_ids = []
        if BOS != None:
            word_ids.append(BOS)
        word_ids.extend(sentence_to_ids(vocab, words)) # list of ids
        if EOS != None:
            word_ids.append(EOS) 

        # ex. word_ids = [BOS, 8, 10, 6, 3, EOS]

        paragraph.append(word_ids) # ex. [[BOS, 8, 10, 6, 3, EOS], [BOS, 5, 13, 9, 7, 25, EOS]]
    
    return paragraph

def drop_paragraphs(paragraphs, max_num_sentences, max_num_words):
    """
    # Arguments:
        paragraphs: list of paragraphs.
        max_num_sentences: int.
        max_num_words: int.
    # Returns:
        paragraphs: list of paragraphs.
    """
    for i, _ in enumerate(paragraphs): # for each paragraph in the list
        paragraphs[i] = paragraphs[i][:max_num_sentences] # drop sentences that are over max_num_sentences
        for j, _ in enumerate(paragraphs[i]): # for each sentence in the given paragraph
            paragraphs[i][j] = paragraphs[i][j][:max_num_words] # drop words that are over max_num_words
    
    return paragraphs

def pad_paragraph(paragraph, max_num_sentences, max_num_words, PAD):
    """
    # Arguments:
        paragraph: list of list of list of ids.
        max_num_sentences: int.
        max_num_words: int.
    # Returns:
        paragraph: list of list of list of ids.
    """
    paragraph.extend([[PAD]] * (max_num_sentences - len(paragraph)))
    for sentence in paragraph:
        sentence.extend([PAD] * (max_num_words - len(sentence)))
    
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
        min_count (optional): int, minimum of word frequency for building vocabrary
        shuffle (optional): bool

    # Params
        PAD, BOS, EOS, UNK: int, id
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN: str
        B, min_count: int
        vocab: Vocab
        word2id: Vocab.word2id
        id2word: Vocab.id2word
        raw_vocab: Vocab.raw_vocab
        V: the size of vocab
        n_data: the number of rows of data
        graph: tf.get_default_graph(), force tf to run model in the same session
        model_s: keras.layer.Model, used to get h_s_pre
        model_w: keras.layer.Model, used to get h_w_pre

    # Examples
        generator = VAESequenceGenerator('./data/train_x.txt', 32)
        x, y_true = generator.__getitem__(idx=11)
        print(x[0])
        >>> 8, 10, 6, 3, 2, 0, 0, ..., 0
        print(y_true[0][0])
        >>> 0, 1, 0, 0, 0, 0, 0, ..., 0

        id2word = generator.id2word

        x_words = [id2word[id] for id in x[0]]
        print(x_words)
        >>> <S> I have a <UNK> </S> <PAD> ... <PAD>
    '''
    def __init__(self, path, B, T, N, min_count=1, shuffle=True):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<S>'
        self.EOS_TOKEN = '</S>'
        self.path = path
        self.B = B
        self.T = T
        self.N = N
        self.min_count = min_count

        default_dict = {
            self.PAD_TOKEN: self.PAD,
            self.BOS_TOKEN: self.BOS,
            self.EOS_TOKEN: self.EOS,
            self.UNK_TOKEN: self.UNK,
        }
        self.vocab = Vocab(default_dict, self.UNK_TOKEN)
        data = load_data(path)
        self.vocab.build_vocab(data, self.min_count)

        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        self.raw_vocab = self.vocab.raw_vocab
        self.V = len(self.vocab.word2id)
        with open(path, 'r', encoding='utf-8') as f:
            self.n_data = sum(1 for line in f)
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
            None: no input is needed for generator pretraining.
            x: numpy.array, shape = (B, T, N)
            y_true: numpy.array, shape = (B, T, N, V)
                labels with one-hot encoding.
                
                T is the max number of sentences in the batch.
                if number of sentences is smaller than max_num_sentences, the data will be padded.

                N is the max number of words in the batch.
                if number of sentences is smaller than max_num_words, the data will be padded.
        '''
        x, y_true = [], []
        start = idx * self.B + 1
        end = (idx + 1) * self.B + 1

        for i in range(start, end):
            if self.shuffle:
                idx = self.shuffled_indices[i]
            else:
                idx = i

            paragraph = linecache.getline(self.path, idx) # str
            paragraph_ids_x = get_paragraph_ids(paragraph, self.vocab, self.BOS, self.EOS)
            paragraph_ids_y_true = get_paragraph_ids(paragraph, self.vocab, EOS=self.EOS)

            x.append(paragraph_ids_x)
            y_true.append(paragraph_ids_y_true)

        x = drop_paragraphs(x, self.T, self.N)
        x = [pad_paragraph(p, self.T, self.N, self.PAD) for p in x]
        x = np.array(x, dtype=np.int32)

        y_true = drop_paragraphs(y_true, self.T, self.N)
        y_true = [pad_paragraph(p, self.T, self.N, self.PAD) for p in y_true]
        y_true = np.array(y_true, dtype=np.int32)
        y_true = to_categorical(y_true, num_classes=self.V)
 
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
            return (x, y_true)

    def __iter__(self):
        return self

    def next(self):
        if self.idx >= self.len:
            self.reset()
            raise StopIteration
        x, y_true = self.__getitem__(self.idx)
        self.idx += 1
        return (x, y_true)

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.shuffled_indices = np.arange(self.n_data)
            random.shuffle(self.shuffled_indices)

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
        min_count (optional): int, minimum of word frequency for building vocabrary
        shuffle (optional): bool

    # Params
        PAD, BOS, EOS, UNK: int, id
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN: str
        B, min_count: int
        vocab: Vocab
        word2id: Vocab.word2id
        id2word: Vocab.id2word
        raw_vocab: Vocab.raw_vocab
        V: the size of vocab
        n_data: the number of rows of data

    # Examples
        generator = VAESequenceGenerator('./data/train_x.txt', 32)
        X, Y = generator.__getitem__(idx=11)
        print(X[0])
        >>> 8, 10, 6, 3, 2, 0, 0, ..., 0
        print(Y)
        >>> 0, 1, 1, 0, 1, 0, 0, ..., 1

        id2word = generator.id2word

        x_words = [id2word[id] for id in X[0]]
        print(x_words)
        >>> I have a <UNK> </S> <PAD> ... <PAD>
    '''
    def __init__(self, path_pos, path_neg, B, T, N, min_count=1, shuffle=True):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<S>'
        self.EOS_TOKEN = '</S>'
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.B = B
        self.T = T
        self.N = N
        self.min_count = min_count

        default_dict = {
            self.PAD_TOKEN: self.PAD,
            self.BOS_TOKEN: self.BOS,
            self.EOS_TOKEN: self.EOS,
            self.UNK_TOKEN: self.UNK,
        }
        self.vocab = Vocab(default_dict, self.UNK_TOKEN)
        data = load_data(path_pos)
        self.vocab.build_vocab(data, self.min_count)

        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        self.raw_vocab = self.vocab.raw_vocab
        self.V = len(self.vocab.word2id)
        with open(path_pos, 'r', encoding='utf-8') as f:
            self.n_data_pos = sum(1 for line in f)
        with open(path_neg, 'r', encoding='utf-8') as f:
            self.n_data_neg = sum(1 for line in f)
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
            Y: numpy.array, shape = (B, )
                labels indicate whether sentences are true data or generated data.
                if true data, y = 1. Else if generated data, y = 0.
        '''
        X, Y = [], []
        start = idx * self.B + 1
        end = (idx + 1) * self.B + 1

        for i in range(start, end):
            idx = self.indicies[i]
            is_pos = 1
            if idx < 0:
                is_pos = 0
                idx = -1 * idx
            idx = idx - 1

            if is_pos == 1:
                paragraph = linecache.getline(self.path_pos, idx) # str
                paragraph_ids_x = get_paragraph_ids(paragraph, self.vocab, EOS=self.EOS)
            elif is_pos == 0:
                paragraph = linecache.getline(self.path_neg, idx) # str
                paragraph_ids_x = get_paragraph_ids(paragraph, self.vocab, EOS=self.EOS)

            X.append(paragraph_ids_x)
            Y.append(is_pos)

        X = drop_paragraphs(X, self.T, self.N)
        X = [pad_paragraph(p, self.T, self.N, self.PAD) for p in X]
        X = np.array(X, dtype=np.int32)

        return (X, Y)

    def __iter__(self):
        return self

    def next(self):
        if self.idx >= self.len:
            self.reset()
            raise StopIteration
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
