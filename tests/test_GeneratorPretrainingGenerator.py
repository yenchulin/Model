from context import unittest, os, GeneratorPretrainingGenerator

top = os.getcwd()

class TestGeneratorPretrainingGenerator(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_generator_pretraining_generator(self):
        gen = GeneratorPretrainingGenerator(
            os.path.join(top, 'data', 'kokoro_parsed.txt'),
            B=8,
            shuffle=False)
        gen.reset()
        x, y_true = gen.next()

        self.assertEqual(y_true.shape[3], gen.V, msg="y shape test")

        """
        A large building with bars on the windows in front of it. (12 words)
        There is people walking in front of the building. (9 words)
        There is a street in front of the building with many cars on it. (14 words)
        """
        expected_text = [
            ['<S>', 'A', 'large', 'building', 'with', 'bars', 'on', 'the', 'windows', 'in', 'front', 'of', 'it.', '</S>', '<PAD>', '<PAD>'], 
            ['<S>', 'There', 'is', 'people', 'walking', 'in', 'front', 'of', 'the', 'building.', '</S>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'],
            ['<S>', 'There', 'is', 'a', 'street', 'in', 'front', 'of', 'the', 'building', 'with', 'many', 'cars', 'on', 'it.', '</S>']
        ] 
        
        num_sentence = len(expected_text)
        max_num_words = 16

        actual_text = []
        for sentence in x[0][:num_sentence]:
            sentence = [gen.id2word[word] for word in sentence[:max_num_words]]
            actual_text.append(sentence)
        
        self.sub_test(actual_text, expected_text, msg='x text test')


        expected_ids = []
        for sentence in expected_text:
            sentence_ids = [gen.word2id[word] for word in sentence]
            expected_ids.append(sentence_ids)

        actual_ids = []
        for sentence in x[0][:num_sentence]:
            sentence = sentence[:max_num_words].tolist()
            actual_ids.append(sentence)
        
        self.sub_test(actual_ids, expected_ids, msg='x ids test')