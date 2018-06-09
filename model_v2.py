import tensorflow as tf
import pickle
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

class Seq2SeqModel:
    def __init__(self, vocab_to_int, batch_size):
        self.learning_rate = 0.001
        self.embedding_size = 128
        self.num_encoder_symbols = len(vocab_to_int)
        self.num_decoder_symbols = len(vocab_to_int)
        self.num_layers = 1
        self.hidden_size = 64
        self.vocab_to_int = vocab_to_int
        self.batch_size = batch_size

    # def set_batch_size(self, batch_size):
    #     self.batch_size = batch_size



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
        return cell

    def encoder_layer_init(self):
        enc_embed_input = tf.contrib.layers.embed_sequence(self.input_data, self.num_encoder_symbols,
                                                           self.embedding_size)
        # enc_embeddings = tf.Variable(tf.random_uniform([self.num_decoder_symbols, self.embedding_size]))
        # enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, self.input_data)
        enc_cell = tf.contrib.rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)])

        enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input,
                                                  sequence_length=self.text_length, dtype=tf.float32)
        return enc_output, enc_state

    def decoding_layer_init(self, enc_state, decoder_input):
        dec_embeddings = tf.Variable(tf.random_uniform([self.num_decoder_symbols, self.embedding_size]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, decoder_input)
        dec_cell = tf.contrib.rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)])
        output_layer = Dense(self.num_decoder_symbols,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        return dec_cell, dec_embed_input, output_layer, dec_embeddings

    def attention_init(self, dec_cell, encoder_outputs, enc_state):
        # attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.hidden_size, memory=encoder_outputs)
        # attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        #     self.hidden_size, attention_states,
        #     memory_sequence_length=self.text_length)
        # decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        #     dec_cell, attention_mechanism,
        #     attention_layer_size=self.hidden_size/2)

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            dec_cell, attention_mechanism,
            attention_layer_size=self.hidden_size)




        return decoder_cell




    def training_helper_init(self, enc_state, output_layer, dec_embed_input, dec_cell):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=self.summary_length,
                                                            time_major=False)
        initial_state = dec_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=enc_state)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, training_helper, initial_state, output_layer)
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

    def inference_helper_init(self, dec_embeddings, output_layer, enc_state, dec_cell):
        start_token = self.vocab_to_int["<GO>"]
        end_token = self.vocab_to_int["<EOS>"]
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [self.batch_size],
                               name='start_tokens')
        initial_state = dec_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=enc_state)
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens, end_token)
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, initial_state, output_layer)
        inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                     output_time_major=False,
                                                                     impute_finished=True,
                                                                     maximum_iterations=self.max_summary_length)[0]
        return inference_decoder_output


    def train_operation_init(self, training_decoder_output, inference_decoder_output):
        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        self.inference_logits = tf.identity(inference_decoder_output.sample_id, name='predictions')

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
        enc_output, enc_state = self.encoder_layer_init()
        decoder_input = self.process_encoding_input(target_data=self.targets)
        dec_cell, dec_embed_input, output_layer, dec_embeddings = self.decoding_layer_init(enc_state=enc_state, decoder_input=decoder_input)
        dec_cell = self.attention_init(dec_cell=dec_cell, encoder_outputs=enc_output, enc_state=enc_state)

        training_logits = self.training_helper_init(enc_state=enc_state, output_layer=output_layer,
                                                  dec_embed_input=dec_embed_input,
                                                  dec_cell=dec_cell)
        # training_logits = self.training_help_init(enc_state=enc_state, output_layer=output_layer, dec_embed_input=dec_embed_input,
        #                                           dec_cell=dec_cell)
        inference_decoder_output = self.inference_helper_init(dec_embeddings=dec_embeddings, output_layer=output_layer,
                                                              enc_state=enc_state, dec_cell=dec_cell)
        self.train_operation_init(training_logits, inference_decoder_output)




def load_data():
    with open('data/vocab_to_int.pkl', 'rb') as f:
        data = pickle.load(f)
    return data
if __name__  == "__main__":

    vocab_to_int = load_data()
    # print(len(vocab_to_int))
    model = Seq2SeqModel(vocab_to_int)
    model.build_model()