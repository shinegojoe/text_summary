import tensorflow as tf
import pickle
from tensorflow.python.layers.core import Dense

class Seq2SeqModel:
    def __init__(self, vocab_to_int):
        self.learning_rate = 0.001
        self.embedding_size = 300
        self.num_encoder_symbols = len(vocab_to_int)
        self.num_decoder_symbols = len(vocab_to_int)
        self.num_layers = 2
        self.hidden_size = 128
        self.vocab_to_int = vocab_to_int
        self.batch_size = 256



    def placeholder_init(self):
        self.input_data = tf.placeholder(dtype=tf.int32, shape=(None, None))
        self.targets = tf.placeholder(dtype=tf.int32, shape=(None, None))
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        self.summary_length = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.max_summary_length = tf.reduce_max(self.summary_length)
        self.text_length = tf.placeholder(dtype=tf.int32, shape=(None,))

    def process_encoding_input(self, target_data):
        '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''

        ending = tf.strided_slice(target_data, [0, 0], [self.batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([self.batch_size, 1], self.vocab_to_int['<GO>']), ending], 1)

        return dec_input

    def make_cell(self):
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        # drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob)
        return cell

    def encoder_layer_init(self):
        enc_embed_input = tf.contrib.layers.embed_sequence(self.input_data, self.num_encoder_symbols,
                                                           self.embedding_size)

        cell_fw = tf.contrib.rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)])
        cell_bw = tf.contrib.rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)])

        # enc_output, enc_state = tf.nn.dynamic_rnn(cell_fw, enc_embed_input,
        #                                           sequence_length=self.text_length, dtype=tf.float32)
        enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                cell_bw,
                                                                enc_embed_input,
                                                                self.text_length,
                                                                dtype=tf.float32)
        # Join outputs since we are using a bidirectional RNN
        # enc_output = tf.concat(enc_output, 2)
        # return enc_state
        return enc_state[0]

    def decoding_layer_init(self, enc_state, decoder_input):
        dec_embeddings = tf.Variable(tf.random_uniform([self.num_decoder_symbols, self.embedding_size]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, decoder_input)
        dec_cell = tf.contrib.rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)])
        output_layer = Dense(self.num_decoder_symbols,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        return dec_cell, dec_embed_input, output_layer

    def training_help_init(self, enc_state, output_layer, dec_embed_input, dec_cell):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=self.summary_length,
                                                            time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, training_helper, enc_state, output_layer)
        # training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True)[0]
        # training_logits, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
        #                                                        output_time_major=False,
        #                                                        impute_finished=True,
        #                                                        maximum_iterations=self.max_summary_length)
        training_logits = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                                maximum_iterations=self.max_summary_length)[0]

        return training_logits

    def train_operation_init(self, training_decoder_output):
        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        masks = tf.sequence_mask(self.summary_length, self.max_summary_length, dtype=tf.float32,
                                 name='masks')
        self.cost = tf.contrib.seq2seq.sequence_loss(training_logits, self.targets, masks)
        # Optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(self.cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)







    def build_model(self):
        self.placeholder_init()
        enc_state = self.encoder_layer_init()
        decoder_input = self.process_encoding_input(target_data=self.targets)
        dec_cell, dec_embed_input, output_layer = self.decoding_layer_init(enc_state=enc_state, decoder_input=decoder_input)
        training_logits = self.training_help_init(enc_state=enc_state, output_layer=output_layer, dec_embed_input=dec_embed_input,
                                                  dec_cell=dec_cell)
        self.train_operation_init(training_logits)




def load_data():
    with open('vocab_to_int.pkl', 'rb') as f:
        data = pickle.load(f)
    return data
if __name__  == "__main__":

    vocab_to_int = load_data()
    # print(len(vocab_to_int))
    model = Seq2SeqModel(vocab_to_int)
    model.build_model()