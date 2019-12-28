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
            T=3,
            N=20,
            vocab=vocab,
            shuffle=False)
        gen.reset()
        x, y = gen.next()

        self.assertEqual(x.shape, (gen.B, gen.N), msg="x shape test")
        self.assertEqual(y.shape, (gen.B, 1), msg="y shape test")

        """
        # 0th sentence: (32 words)
        i do not play it but i was pleased to know that this does not get overheated like the other one and it does not sound super loud like the other one.
        """
        expected_text = ['i', 'do', 'not', 'play', 'it', 'but', 'i', 'was', 'pleased', 'to', 'know', 'that', 'this', 'does', 'not', 'get', 'overheated', 'like', 'the', '</S>']

        actual_text = [vocab.id2word[_id] for _id in x[0]]
        self.sub_test(actual_text, expected_text, msg='x positive text test - 1 (original text > 20 words)')
        self.sub_test(y[0], [1], msg='true data')

        """
        # 2th sentence: (7 words)
        ps4 pro is a total entertainment machine!
        """
        expected_text = ['ps4', 'pro', 'is', 'a', 'total', 'entertainment', 'machine', '</S>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']

        actual_text = [vocab.id2word[_id] for _id in x[2]]
        self.sub_test(actual_text, expected_text, msg='x positive text test - 2 (original text < 20 words)')
        self.sub_test(y[2], [1], msg='true data')


        x, y = gen.__getitem__(idx=(gen.n_data_pos // gen.B + 1))
        expected_text = ['possible', '<PAD>', 'has', 'i', 'but', '4k', 'will', '<PAD>', 'only', '<PAD>', 'i', 'no', 'not', 'it', '</S>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '</S>']

        actual_text = [vocab.id2word[_id] for _id in x[0]]
        self.sub_test(actual_text, expected_text, msg='x neg text test')
        self.sub_test(y[0], [0], 'generated data')