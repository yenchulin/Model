from tests.context import unittest, os, DiscriminatorSentenceGenerator, Vocab

top = os.getcwd()

class TestDiscriminatorSentenceGenerator(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_discriminator_generator(self):
        path_pos = os.path.join(top, 'data', 'kokoro_parsed_sentence.txt')
        path_neg = os.path.join(top, 'tests', 'data', 'generated_sentences.txt')
        vocab = Vocab(path_pos)
        gen = DiscriminatorSentenceGenerator(
            path_pos=path_pos,
            path_neg=path_neg,
            B=8,
            T=5,
            N=20,
            vocab=vocab,
            shuffle=False)
        gen.reset()
        x, y = gen.next()

        self.assertEqual(x.shape, (gen.B, gen.N), msg="x shape test")
        self.assertEqual(y.shape, (gen.B, 1), msg="y shape test")

        """
        A large building with bars on the windows in front of it. (12 words)
        There is people walking in front of the building. (9 words)
        There is a street in front of the building with many cars on it. (14 words)
        """
        expected_text = [
            ['a', 'large', 'building', 'with', 'bars', 'on', 'the', 'windows', 'in', 'front', 'of', 'it', '</S>', '<PAD>', '<PAD>', '<PAD>','<PAD>', '<PAD>', '<PAD>', '<PAD>'], 
            ['there', 'is', 'people', 'walking', 'in', 'front', 'of', 'the', 'building', '</S>', '<PAD>', '<PAD>', '<PAD>','<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'],
            ['there', 'is', 'a', 'street', 'in', 'front', 'of', 'the', 'building', 'with', 'many', 'cars', 'on', 'it', '</S>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'],
            ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>','<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'],
            ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>','<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
        ] 

        actual_text = [vocab.id2word[word] for word in x[0]]
        self.sub_test(actual_text, expected_text, msg='x pos text test')
        self.sub_test(y[0][0], [1], msg='true data')


        x, y = gen.__getitem__(idx=(gen.n_data_pos // gen.B + 1))
        expected_text = [
            ['underneath', 'partially', 'this', '<PAD>', '<PAD>', 'a', 'and', 'and', '<PAD>', 'white', 'wears', '<PAD>', 'in', '<PAD>', 'black', '<PAD>', 'wearing', 'window', 'fence', 'and'],
            ['are', '</S>', 'climbing', 'baby', '<PAD>', 'bathing', '<PAD>', '<PAD>', '<PAD>', 'his', 'a', 'are', '<PAD>', 'a', 'it', '<PAD>', 'rail', 'the', '<PAD>', 'exposed'],
            ['white', 'dog', '<PAD>', '<PAD>', 'red', '<PAD>', 'above', 'shirt', '<PAD>', '<PAD>', '<PAD>', '<PAD>', 'of', 'sink', '<PAD>', '<PAD>', 'been', 'with', 'in', 'thin'],
            ['<PAD>', '<PAD>', '<PAD>', 'inside', 'tennis', 'which', '<PAD>', '<PAD>', 'a', '<PAD>', 'yellow', 'with', 'yellow', '<PAD>', 'is', 'the', '<PAD>', 'tub', 'two', 'a'],
            ['<PAD>', '<PAD>', 'in', '<PAD>', 'many', 'there', 'hair', 'right', '<PAD>', 'is', '<PAD>', 'in', '<PAD>', '<PAD>', '<PAD>', 'beside', '<PAD>', 'and', 'at', 'appears']
        ]

        actual_text = []
        for sentence_ids in x[0]:
            sentence = [vocab.id2word[_id] for _id in sentence_ids]
            actual_text.append(sentence)
        
        self.sub_test(actual_text, expected_text, msg='x neg text test')
        self.sub_test(y[0][0], [0], 'generated data')