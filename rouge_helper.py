import numpy as np
from nltk.translate.bleu_score import sentence_bleu

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
            # score = sentence_bleu([ref], sentence)
            batch_score.append(score)

        mean_score = np.mean(batch_score)
        return mean_score

def remove_pad(padded):
    pad_int = 62718
    sentences = []
    for line in padded:
        line_buf = []
        for word in line:
            if word != pad_int:
                line_buf.append(word)
        sentences.append(line_buf)
    return sentences

def main():
    # h = ROUGEHelper()
    s = ['the', 'cat', 'was', 'found', 'under', 'the', 'bed']
    r = ['the', 'cat', 'was', 'under', 'the', 'bed']
    # s2 = [9049, 21128, 22556, 5055, 62718, 62718, 62718]
    # r2 = [9049, 21128, 22556,  5055, 62718, 62718, 62718]
    # S = [s, s2]
    # R = [r, r2]
    # # score = h.get_socre(s, r, 2)
    # score = h.get_batch_score(S, R)
    # print(score)
    # test = [[53695, 40358, 59067,  5055, 62718, 62718],
    #  [44246, 45763, 62718, 62718, 62718, 62718]]

    # qq = remove_pad(test)
    # for q in qq:
    #     print(q)
    reference = [['The', 'cat', 'is', 'on', 'the', 'mat']]
    candidate = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    score = sentence_bleu(reference, candidate)
    print(score)



if __name__== "__main__":
    main()