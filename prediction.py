import tensorflow as tf
import pickle
from model_v2 import Seq2SeqModel

def load_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def load_data():
    base_path = 'data/'
    # summaries = load_pkl(base_path + 'sorted_summaries')
    # texts = load_pkl(base_path + 'sorted_texts')
    summaries = load_pkl(base_path + 'min_summaries')
    texts = load_pkl(base_path + 'min_texts')
    vocab_to_int = load_pkl(base_path + 'vocab_to_int.pkl')

    return summaries, texts, vocab_to_int

def id_to_vocab(data, int_to_vovab):
    sentence = []
    for id in data:
        word = int_to_vovab[id]
        sentence.append(word)

    return sentence


def prediction():
    summaries, texts, vocab_to_int = load_data()
    int_to_vocab = load_pkl('data/' + 'int_to_vocab.pkl')
    print(int_to_vocab)


    model = Seq2SeqModel(vocab_to_int)
    # model.set_batch_size(1)
    model.build_model()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "my_net/save_net.ckpt")
        # model.set_batch_size(1)
        text_id = texts[10]
        sum_id = summaries[10]

        # text = int_to_vocab(ids=text_id, int_to_vocab=int_to_vocab)
        # sumXX = int_to_vocab(sum_id, int_to_vocab)
        # print(text_id)
        # print(sum_id)
        batch_size = 256
        feed = {model.input_data: [text_id] * batch_size,
                model.summary_length: [len(text_id)] * batch_size,
                model.text_length: [len(text_id)] * batch_size
                }
        inference_logitis = sess.run(model.inference_logits, feed_dict=feed)[0]
        print('inference_logitis', inference_logitis)


        text = id_to_vocab(text_id, int_to_vocab)
        summary = id_to_vocab(sum_id, int_to_vocab)
        predic = id_to_vocab(inference_logitis, int_to_vocab)

        print(text)
        print(summary)
        print(predic)


prediction()