import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper

from utils.tf.tf_helpers import tf_helpers
import logging
class StoryClozeLSTMReader(object):
    def __init__(self,
                 n_classes,
                 embeddings,
                 embeddings_size,
                 embeddings_number,
                 hidden_size,
                 learn_rate=0.001,
                 loss_l2_beta=0.01,
                 layer_out_noise=0.00,
                 learning_rate_trainable=False,
                 embeddings_trainable=True,
                 bilstm_layers=1,
                 dropout_prob_keep=0.00):


        # input params
        """
        """
        self.l2_beta = loss_l2_beta
        self.learn_rate = learn_rate

        self.embeddings_size = embeddings_size if embeddings is None else len(embeddings[0])
        self.embeddings_number = embeddings_number if embeddings is None else len(embeddings)
        self.embeddings_trainable = embeddings_trainable

        self.n_classes = n_classes
        self.hidden_size = hidden_size

        # input params
        self.input_story_tokens = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_story_tokens")
        self.input_answer1_tokens = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_answer1_tokens")
        self.input_answer2_tokens = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_answer2_tokens")

        self.input_story_tokens_seq_len = tf.placeholder(dtype=tf.int32, shape=[None],
                                                         name="input_story_tokens_seq_len")  # length of every sequence in the batch
        self.input_answer1_tokens_seq_len = tf.placeholder(dtype=tf.int32, shape=[None],
                                                         name="input_story_tokens_seq_len")  # length of every sequence in the batch
        self.input_answer2_tokens_seq_len = tf.placeholder(dtype=tf.int32, shape=[None],
                                                         name="input_story_tokens_seq_len")  # length of every sequence in the batch

        # self.input_y = tf.placeholder(dtype=tf.int32, shape=[None], name="input_y")  # this is not onehot!

        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None], name="input_y")  # this is onehot!
        self.input_y_onehot = tf.one_hot(self.input_y, n_classes)

        # model settings
        self.weights = {}
        self.biases = {}

        with tf.name_scope("embeddings"):
            # embeddings_placeholder = tf.placeholder(tf.float32, shape=[embeddings_number, embedding_size])
            # embeddings_tuned = tf.Variable(embeddings_placeholder)

            # embeddings random, tuned
            # embeddings_tuned =tf.Variable(tf.truncated_normal([embeddings_number, embedding_size], stddev=0.1, dtype=tf.float32), trainable=False, name="embeddings", dtype=tf.float32)

            # embeddings, loaded, tuned
            if not embeddings is None:
                self.embeddings_tuned = tf.Variable(embeddings,
                                               trainable=self.embeddings_trainable,
                                               name="embeddings",
                                               dtype=tf.float32)
            else:
                self.embeddings_tuned = tf.Variable(
                                                tf.truncated_normal([self.embeddings_number, self.embeddings_size],
                                                                    stddev=0.1,
                                                                    dtype=tf.float32),
                                                trainable=self.embeddings_trainable,
                                                name="embeddings", dtype=tf.float32)

            self.embedded_chars_story = tf.nn.embedding_lookup(self.embeddings_tuned, self.input_story_tokens)
            self.embedded_chars_answer1 = tf.nn.embedding_lookup(self.embeddings_tuned, self.input_answer1_tokens)
            self.embedded_chars_answer2 = tf.nn.embedding_lookup(self.embeddings_tuned, self.input_answer2_tokens)

            # embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)v# not required for rnn/lstms

        # def BiLSTM_dynamic(x_embedd, input_seq_len, shared_fw_bw=False,
        #                    use_peepholes=False, cell_clip=None,
        #                    # initializer=None,
        #                    num_proj=None, proj_clip=None,
        #                    num_unit_shards=1, num_proj_shards=1,
        #                    forget_bias=1.0, state_is_tuple=True,
        #                    activation=tf.tanh,
        #                    scope_id=0,
        #                    lstm_layers=1):
        #
        #     with tf.variable_scope("BiLSTM-base"):
        #         with tf.variable_scope('forward-%s' % scope_id, reuse=True):
        #             cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True,
        #                                               use_peepholes=use_peepholes, cell_clip=cell_clip,
        #                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=42),
        #                                               num_proj=num_proj, proj_clip=proj_clip,
        #                                               num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
        #                                               forget_bias=forget_bias,
        #                                               activation=activation)
        #
        #         with tf.variable_scope('backward-%s' % scope_id, reuse=True):
        #             cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True,
        #                                               use_peepholes=use_peepholes, cell_clip=cell_clip,
        #                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=24),
        #                                               num_proj=num_proj, proj_clip=proj_clip,
        #                                               num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
        #                                               forget_bias=forget_bias,
        #                                               activation=activation)
        #
        #         multicell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * lstm_layers, state_is_tuple=True)
        #         multicell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * lstm_layers, state_is_tuple=True)
        #         outputs, states = tf.nn.bidirectional_dynamic_rnn(
        #             cell_fw=multicell_fw,
        #             cell_bw=multicell_fw if shared_fw_bw else multicell_bw,
        #             dtype=tf.float32,
        #             sequence_length=input_seq_len,
        #             inputs=x_embedd)
        #
        #     return outputs, states

        # Dynamic BiLSTM outputs and states

        def tf_gather_on_axis1_draft(tensor, indices, axis=1):
            axis = 1
            # gather with gradients support
            # http://stackoverflow.com/questions/36088277/how-to-select-rows-from-a-3-d-tensor-in-tensorflow
            # tensor shape=[50, 55, 20]
            # indices: 50 items [2,5,34,54,...,]

            n_indices = tf.shape(indices)[0] # number of indices 50
            #tensor_flat = tf.reshape(tensor, [axis]) # shape=[50*55, 20]
            # tensor_flat = tf.split(0, tensor_shape[0]+tensor_shape[1], tf.reshape(tensor, [-1]))
            # tensor_flat = tf.reshape(tensor, [tensor_shape[0] * tensor_shape[1], tensor_shape[1]])
            tensor_flat = tf.reshape(tensor, tf.concat(0,[tf.expand_dims(tf.reduce_prod(tf.shape(tensor)[:axis+1]),0),tf.shape(tensor)[axis+1:]]))

            index_offset = tf.range(0, tf.reduce_prod(tf.shape(tensor)[:axis+1]), tf.shape(tensor)[axis]) # shape=[50,1], tensor [[55],[110],165....]
            flattened_indices = indices+index_offset

            res = tf.gather(tensor_flat, flattened_indices)

            return res

        def tf_gather_on_axis1(tensor, indices, axis=1):
            tensor_flat = tf.reshape(tensor, tf.concat(0,
                                                       [tf.expand_dims(tf.reduce_prod(tf.shape(tensor)[:axis + 1]), 0),
                                                        tf.shape(tensor)[axis + 1:]]))
            index_offset = tf.range(0, tf.reduce_prod(tf.shape(tensor)[:axis+1]), tf.shape(tensor)[axis]) # shape=[50,1], tensor [[55],[110],165....]
            flattened_indices = indices+index_offset
            res = tf.gather(tensor_flat, flattened_indices)
            return res

        lstm_layers_cnt = bilstm_layers
        shared_fw_bw = False
        with tf.variable_scope('lstm-encode') as lstm_encode_scope:
            def encode_sequence(x_embedd, input_seq_len, lstm_cell_fw, lstm_cell_bw, initial_state=None,
                                bidirectional=False):
                output_layer_story = None
                output_states_story = None
                if bidirectional:
                    outputs_story, states_story = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=lstm_cell_fw,
                        cell_bw=lstm_cell_bw,
                        dtype=tf.float32,
                        sequence_length=input_seq_len,
                        inputs=x_embedd)

                    output_layer_story = tf.concat(2, outputs_story, name="output_layer")  # concat fw and bw layers
                    output_states_story = states_story
                else:
                    outputs_story, states_story = tf.nn.dynamic_rnn(
                        cell=lstm_cell_fw,
                        dtype=tf.float32,
                        sequence_length=input_seq_len,
                        inputs=x_embedd,
                        initial_state=initial_state)

                    output_layer_story = outputs_story  # concat fw and bw layers
                    output_states_story = states_story

                return output_layer_story, output_states_story

            lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
                                                use_peepholes=False,
                                                cell_clip=None,
                                                num_proj=None,
                                                proj_clip=None,
                                                num_unit_shards=1, num_proj_shards=1,
                                                forget_bias=1.0,
                                                activation=tf.tanh,
                                                state_is_tuple=True,
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=42))


            lstm_cell_fw = DropoutWrapper(lstm_cell_fw, output_keep_prob=dropout_prob_keep)

            # multicell_fw = lstm_cell_fw# tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * lstm_layers_cnt, state_is_tuple=True)

            # if not shared_fw_bw:
            #     lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,
            #                                            use_peepholes=False,
            #                                            cell_clip=None,
            #                                            num_proj=None,
            #                                            proj_clip=None,
            #                                            num_unit_shards=1, num_proj_shards=1,
            #                                            forget_bias=1.0,
            #                                            activation=tf.tanh,
            #                                            state_is_tuple=True,
            #                                            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=42))
            # else:
            #     lstm_cell_bw = lstm_cell_fw
            #
            #     multicell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * lstm_layers_cnt, state_is_tuple=True)

            multicell_bw = None

            output_layer = None
            output_layer_size = 0

            # story encoding
            # story encoding
            bidirectional = False

            output_layer_size = 2 * hidden_size if bidirectional else hidden_size

            # shortcut - use POS and deps embeddings on the output layer

            encoded_story, encoded_story_states = encode_sequence(self.embedded_chars_story, self.input_story_tokens_seq_len, lstm_cell_fw=lstm_cell_fw, lstm_cell_bw=multicell_bw)
            self.encoded_story_states = encoded_story_states
            encoded_story_last_output = tf_gather_on_axis1(encoded_story, self.input_story_tokens_seq_len - tf.ones_like(self.input_story_tokens_seq_len))
            logging.info("encoded_story_last_output shape: %s" % (encoded_story_last_output.get_shape()))

            lstm_encode_scope.reuse_variables()
            encoded_answer1, _ = encode_sequence(self.embedded_chars_answer1, self.input_answer1_tokens_seq_len, lstm_cell_fw=lstm_cell_fw, lstm_cell_bw=multicell_bw, initial_state=encoded_story_states)
            encoded_answer2, _ = encode_sequence(self.embedded_chars_answer2, self.input_answer2_tokens_seq_len, lstm_cell_fw=lstm_cell_fw, lstm_cell_bw=multicell_bw, initial_state=encoded_story_states)

            encoded_answer1_last_output = tf_gather_on_axis1(encoded_answer1, self.input_answer1_tokens_seq_len - tf.ones_like(self.input_answer1_tokens_seq_len))
            encoded_answer2_last_output = tf_gather_on_axis1(encoded_answer2, self.input_answer2_tokens_seq_len - tf.ones_like(self.input_answer2_tokens_seq_len))


        # all_feats = tf.concat(1, [encoded_story_last_output, encoded_answer1_last_output, encoded_answer2_last_output])
        all_feats = tf.concat(1, [encoded_answer1_last_output, encoded_answer2_last_output])

        logging.info("all_feats:%s " % all_feats.get_shape())
        output_layer_size = 2 * output_layer_size

        # Calculate logits and probs
        # Reshape so we can calculate them all at once
        # bi_output_concat_flat = tf.reshape(output_layer, [-1, output_layer_size])
        # bi_output_concat_flat_clipped = tf.clip_by_value(bi_output_concat_flat, -1., 1.)  # use clipping if needed. Currently not


        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        self.weights['out'] = tf.Variable(tf.random_uniform([output_layer_size, n_classes], minval=-0.1, maxval=0.1, dtype=tf.float32), name="out_w", dtype=tf.float32)
        self.biases['out'] = tf.Variable(tf.random_uniform([n_classes], minval=-0.1, maxval=0.1, dtype=tf.float32), name="out_b", dtype=tf.float32)

        self.logits = tf.matmul(all_feats, self.weights["out"]) + self.biases["out"]

        # self.logits_flat = tf.batch_matmul(bi_output_concat_flat, self.weights["out"]) + self.biases["out"]
        # logits_flat = tf.clip_by_value(logits_flat, -1., 1.)  # use clipping if needed. Currently not
        self.probs = tf.nn.softmax(self.logits)

        self.predictions = tf.argmax(self.probs, 1, name="predictions")

        # print "outputs[-1].shape:%s" % outputs[-1]  #
        logging.info("weights[\"out\"]].shape:%s" % self.weights["out"])  # .shape
        logging.info("biases[\"out\"].shape:%s" % self.biases["out"])  # .shape

        # print pred

        # make y flat so it match pred shape
        # self.input_y_flat = tf.reshape(self.input_y, [-1])
        # print "input_y_flat:%s" % self.input_y_flat

        # # Define loss and optimizer

        # sparse_softmax_cross_entropy_with_logits - calculates on non-onehot y!
        # this should also not calculate the cross entropy for 0 labels (the padded labels)
        self.losses = tf_helpers.tf_nan_to_zeros_float32(
            tf.nn.softmax_cross_entropy_with_logits(self.logits, self.input_y_onehot))

        # Replace nans with zeros - yep, there are some Nans from time to time..
        # self.losses = self.losses)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # Applying the gradients is outside this class!
        # vars = tf.trainable_variables()
        # loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.l2_beta

        self.mean_loss = tf.reduce_mean(self.losses)

        # Compute and apply gradients
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)

        gvs = optimizer.compute_gradients(self.mean_loss)

        logging.info("gradients:")
        for grad, var in gvs:
            logging.info("%s - %s" % (grad, var))
        capped_gvs = [(tf.clip_by_value(tf_helpers.tf_nan_to_zeros_float32(grad), -6.,
                                        6.) if grad is not None else grad, var) for grad, var in
                      gvs]  # cap to prevent NaNs

        self.apply_grads_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

        logging.info("The model might be okay :D")
        pass




