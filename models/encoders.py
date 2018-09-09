import tensorflow as tf
import logging


def encode_sequence(x_embedd, input_seq_len, lstm_cell_fw, lstm_cell_bw, initial_state=None,
                    bidirectional=False,
                    concat_outputs=True):
    output_layer_story = None
    output_states_story = None
    if bidirectional:
        outputs_story, states_story = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_cell_fw,
            cell_bw=lstm_cell_bw,
            dtype=tf.float32,
            sequence_length=input_seq_len,
            inputs=x_embedd,
            initial_state_fw=initial_state[0] if initial_state is not None else None,
            initial_state_bw=initial_state[1] if initial_state is not None else None)

        if concat_outputs:
            output_layer_story = tf.concat(2, outputs_story)  # concat fw and bw layers
        else:
            output_layer_story = outputs_story
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


def encode_sequence_rank4_0bs_1factcnt_2seqlen_3dim(rank4_seq, rank3_seq_lens, lstm_cell_fw_knw,
                                                    lstm_cell_bw_knw, initial_state,
                                                    bidirectional_knowledge,
                                                    word_emb_knowledge_size,
                                                    know_hidden_size
                                                    ):
    # logging.info(tf.shape(rank4_seq))
    know_data_shape = tf.shape(rank4_seq)  # b, facts_cnt, l, k
    know_data_reshape = tf.reshape(rank4_seq,
                                   shape=[know_data_shape[0] * know_data_shape[1], know_data_shape[2],
                                          know_data_shape[3]])  # b * facts_cnt, l, k
    know_data_reshape.set_shape([None, None,
                                 word_emb_knowledge_size])  # Fix suggested in https://github.com/tensorflow/tensorflow/issues/2938

    know_data_seq_len_reshape = tf.reshape(rank3_seq_lens, [-1])

    encoded_knowledge_subj, encoded_knowledge_subj_states = encode_sequence(know_data_reshape,
                                                                            know_data_seq_len_reshape,
                                                                            lstm_cell_fw=lstm_cell_fw_knw,
                                                                            lstm_cell_bw=lstm_cell_bw_knw,
                                                                            initial_state=initial_state,
                                                                            bidirectional=bidirectional_knowledge)

    logging.info("encoded_knowledge_subj_states: %s" % str(encoded_knowledge_subj_states))

    encoded_knowledge_size = know_hidden_size * (2 if bidirectional_knowledge else 1)
    encoded_knowledge_subj_last = tf.reshape(tf.concat(1,
                                                       encoded_knowledge_subj_states) if bidirectional_knowledge else encoded_knowledge_subj_states,
                                             [know_data_shape[0], know_data_shape[1],
                                              encoded_knowledge_size])

    return encoded_knowledge_subj_last, encoded_knowledge_subj_states
