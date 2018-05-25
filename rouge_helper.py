import numpy as np

class ROUGEHelper():
    def __init__(self):
        pass

    def get_socre(self, sentence, ref, window_length):
        # window_length =

        reference_gram = []
        sentence_gram = []

        reference_end_index = len(ref) - window_length + 1
        sentence_end_index = len(sentence) - window_length + 1

        for i in range(reference_end_index):
            gram_buf = []
            gram_buf.append(ref[i])
            gram_buf.append(ref[i + 1])
            reference_gram.append(gram_buf)

        for i in range(sentence_end_index):
            gram_buf = []
            gram_buf.append(sentence[i])
            gram_buf.append(sentence[i + 1])
            sentence_gram.append(gram_buf)

        # print(reference_gram)
        # print(sentence_gram)

        common_count = 0
        reference_count = len(reference_gram)

        for gram in sentence_gram:
            if gram in reference_gram:
                common_count += 1

        # for word in s1:
        #     if word in r1:
        #         common_count += 1
        #
        score = common_count / reference_count
        return score

    def get_batch_score(self, batch_sentence, batch_ref):
        batch_score = []
        batch_size = len(batch_sentence)
        for i in range(batch_size):
            sentence = batch_sentence[i]
            ref = batch_ref[i]
            # sentence_words = sentence.split()
            # ref_words = ref.split()

            sentence_words = sentence
            ref_words = ref

            score = self.get_socre(sentence=sentence_words, ref=ref_words, window_length=2)
            batch_score.append(score)

        mean_score = np.mean(batch_score)
        return mean_score