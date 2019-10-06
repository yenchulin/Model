from context import unittest, os, DiscriminatorGenerator

top = os.getcwd()

class TestGeneratorPretrainingGenerator(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_generator_pretraining_generator(self):
        gen = DiscriminatorGenerator(
            path_pos=os.path.join(top, 'data', 'kokoro_parsed.txt'),
            path_neg=os.path.join(top, 'data', 'kokoro_parsed.txt'),
            B=1,
            T=4,
            N=10,
            shuffle=False)
        gen.reset()
        x, y = gen.next()
        
        """
        A large building with bars on the windows in front of it. (12 words)
        There is people walking in front of the building. (9 words)
        There is a street in front of the building with many cars on it. (14 words)
        """
        expected_text = [
            ['A', 'large', 'building', 'with', 'bars', 'on', 'the', 'windows', 'in', 'front'], 
            ['There', 'is', 'people', 'walking', 'in', 'front', 'of', 'the', 'building.', '</S>'],
            ['There', 'is', 'a', 'street', 'in', 'front', 'of', 'the', 'building', 'with'],
            ['<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
        ] 

        actual_text = []
        for sentence in x[0]:
            sentence = [gen.id2word[word] for word in sentence]
            actual_text.append(sentence)

        self.sub_test(actual_text, expected_text, msg='x text test')
        self.sub_test(y[0], 1, msg='true data')


        x, y = gen.__getitem__(idx=gen.n_data_pos)
        expected_ids = []
        for sentence in expected_text:
            sentence_ids = [gen.word2id[word] for word in sentence]
            expected_ids.append(sentence_ids)

        actual_ids = x[0].tolist()
        
        self.sub_test(actual_ids, expected_ids, msg='x ids test')
        self.sub_test(y[0], 0, 'generated data')