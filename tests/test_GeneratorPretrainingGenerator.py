from tests.context import unittest, os, GeneratorPretrainingGenerator, Vocab

top = os.getcwd()

class TestGeneratorPretrainingGenerator(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_generator_pretraining_generator(self):
        file_path = os.path.join(top, 'data', 'kokoro_parsed.txt')
        vocab = Vocab(file_path)
        gen = GeneratorPretrainingGenerator(
            path=file_path,
            B=8,
            T=5,
            N=20,
            vocab=vocab,
            shuffle=False)
        gen.reset()
        x, y_true, loss_weight = gen.next()

        self.assertEqual(x[0].shape, (gen.B, gen.T, gen.N), msg="x shape test")
        self.assertEqual(y_true.shape, (gen.B, gen.T, gen.N, vocab.V), msg="y shape test")

        """
        A large building with bars on the windows in front of it. (12 words)
        There is people walking in front of the building. (9 words)
        There is a street in front of the building with many cars on it. (14 words)
        """
        expected_text = [
            ['<S>', 'a', 'large', 'building', 'with', 'bars', 'on', 'the', 'windows', 'in', 'front', 'of', 'it', '</S>', '<PAD>', '<PAD>', '<PAD>','<PAD>', '<PAD>', '<PAD>'], 
            ['<S>', 'there', 'is', 'people', 'walking', 'in', 'front', 'of', 'the', 'building', '</S>', '<PAD>', '<PAD>', '<PAD>','<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'],
            ['<S>', 'there', 'is', 'a', 'street', 'in', 'front', 'of', 'the', 'building', 'with', 'many', 'cars', 'on', 'it', '</S>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'],
            ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>','<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'],
            ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>','<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
        ] 

        actual_text = []
        for sentence in x[0][0]:
            sentence = [vocab.id2word[word] for word in sentence]
            actual_text.append(sentence)
        self.sub_test(actual_text, expected_text, msg='x text test')


        expected_ids = []
        for sentence in expected_text:
            sentence_ids = [vocab.word2id[word] for word in sentence]
            expected_ids.append(sentence_ids)
        actual_ids = x[0][0].tolist() 
        self.sub_test(actual_ids, expected_ids, msg='x ids test')

        expected_loss_weight = [14, 11, 16, 0, 0]
        actual_loss_weight = loss_weight[0].tolist()
        self.sub_test(actual_loss_weight, expected_loss_weight, msg='x loss_weoght test')