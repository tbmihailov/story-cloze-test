import tensorflow as tf
import logging
class tf_helpers(object):

    @staticmethod
    def get_rnn_cell(rnn_cell_type, hidden_size, initializer):
        """Returns LSTM or GRU cell"""
        rnn_cell = None
        if rnn_cell_type.lower() == "lstm":
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, initializer=initializer)
        elif rnn_cell_type.lower() == "gru":
            rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size, activation=tf.tanh)
        else:
            raise Exception("RNN cell type %s is not supported!" % rnn_cell_type)

        return rnn_cell

    @staticmethod
    def tf_nan_to_zeros_float64(tensor):
        """
            Mask NaN values with zeros
        :param tensor: Tensor that might have Nan values
        :return: Tensor with replaced Nan values with zeros
        """
        return tf.select(tf.is_nan(tensor), tf.zeros(tf.shape(tensor), dtype=tf.float64), tensor)

    @staticmethod
    def tf_nan_to_zeros_float32(tensor):
        """
            Mask NaN values with zeros
        :param tensor: Tensor that might have Nan values
        :return: Tensor with replaced Nan values with zeros
        """
        return tf.select(tf.is_nan(tensor), tf.zeros(tf.shape(tensor), dtype=tf.float32), tensor)

    @staticmethod
    def tf_zeros_to_one_float32_multi_dim(tensor):
        """
            Mask zero values with one. TO be used when divide by zeros
        :param tensor: Tensor that might have zero values
        :return: Tensor with replaced zero values with ones
        """

        tshape = tf.shape(tensor)
        tensor_reshaped = tf.reshape(tensor, shape=[-1])
        res = tf.select(tensor_reshaped < 0.001, tf.ones(tf.shape(tensor_reshaped), dtype=tf.float32), tensor_reshaped)
        res_reshaped = tf.reshape(res, shape=tshape)

        return res_reshaped

    @staticmethod
    def tf_matmul_broadcast(tensor1_ijk, tensor2_kl):
        i = tf.shape(tensor1_ijk)[0]
        j = tf.shape(tensor1_ijk)[1]
        k = tf.shape(tensor1_ijk)[2]

        l = tf.shape(tensor2_kl)[1]

        a_flat = tf.reshape(tensor1_ijk, [-1, k])
        res_flat = tf.matmul(a_flat, tensor2_kl)

        res_desired = tf.reshape(res_flat, tf.pack([i, j, l]))

        return res_desired

    @staticmethod
    def tf_gather_on_axis1(tensor, indices, axis=1):
        tensor_flat = tf.reshape(tensor, tf.concat(0,
                                                   [tf.expand_dims(tf.reduce_prod(tf.shape(tensor)[:axis + 1]), 0),
                                                    tf.shape(tensor)[axis + 1:]]))
        index_offset = tf.range(0, tf.reduce_prod(tf.shape(tensor)[:axis + 1]),
                                tf.shape(tensor)[axis])  # shape=[50,1], tensor [[55],[110],165....]
        flattened_indices = indices + index_offset
        res = tf.gather(tensor_flat, flattened_indices)
        return res

    @staticmethod
    def tf_gather_3dims_from_2_dims(tensor_batch_values, tensor_batch_indices):
        """
        Gather values from batched data values [bs, l], indices [bs, l1, l2], gathering the values from l2 from the items from l.
        :param tensor_batch_values: Tensor of floats with size [bs, l]
        :param tensor_batch_indices: Tensor of int indices with size indices [bs, l1, l2]. Indices in dim 3 are indeces in [0,l-1]
        :return: Gathered values from tensor_batch_values and tb
        """
        axis = 1

        tensor_values_flat = tf.reshape(tensor_batch_values, [-1])  # Make one dimensional - to work with gather
        indices_reduce_dime = tf.reshape(tensor_batch_indices, [tf.shape(tensor_batch_indices)[0],
                                                                -1])  # Make indices from rank 3 to rank 2 - per item in batch

        index_offset = tf.expand_dims(tf.range(0, tf.reduce_prod(tf.shape(tensor_batch_values)[:axis + 1]),
                                               tf.shape(tensor_batch_values)[axis]),
                                      -1)  # Generate offests for every item in the batch

        indices_with_offset = indices_reduce_dime + index_offset  # Add the offset to the indices per item in the batch

        indices_with_offset_flat = tf.reshape(indices_with_offset, [-1])  # Make flat to gather

        gathered_values = tf.gather(tensor_values_flat, indices_with_offset_flat)  # Gather values from flat

        gathered_values_reshape = tf.reshape(gathered_values, tf.shape(tensor_batch_indices))

        return gathered_values_reshape


    @staticmethod
    def tf_gather_batch_items(tensor_batch_values,
                              tensor_batch_indices):
        """
        For each batch id take the nth elem from the batch items
        :param tensor_batch_values: Tensor of floats with size [bs, items, hs]
        :param tensor_batch_indices: Tensor of int indices with size indices [bs, l1]. 
        :return: Gathered values from tensor_batch_values : from items get l1 indices
        """
        # tensor_values_flat = tf.reshape(tensor_batch_values, [-1, tf.shape(tensor_batch_values)[-1], ])  # Make one dimensional - to work with gather
        # indices_reduce_dime = tensor_batch_indices  # Make indices from rank 3 to rank 2 - per item in batch

        batch_idx = tf.range(0, tf.shape(tensor_batch_values)[0])  # 0 to batch size -1   [0,1,2,3]
        batch_idx = tf.expand_dims(batch_idx, -1)  # [[0],[1],[2],[3]] , [bs, 1]

        batch_idx_tiled = tf.tile(batch_idx, [1, tf.shape(tensor_batch_indices)[1]])  # [bs, indices_num, 1]
        batch_idx_tiled = tf.expand_dims(batch_idx_tiled, -1)
        logging.info("batch_idx_tiled:%s" % str(batch_idx_tiled))

        tensor_batch_indices_expand = tf.expand_dims(tensor_batch_indices, -1)
        logging.info("tensor_batch_indices_expand:%s" % str(tensor_batch_indices_expand))

        tensor_batch_indices_batched = tf.concat(2, [batch_idx_tiled, tensor_batch_indices_expand])

        gathered_values = tf.gather_nd(tensor_batch_values, tensor_batch_indices_batched)

        return gathered_values

    @staticmethod
    def tf_gather_on_last_axis_2d(batch_tensor_2d, batch_indices_2d):
        tensor_flat = tf.reshape(batch_tensor_2d, [-1])

        index_offset = tf.expand_dims(
            tf.range(0, tf.reduce_prod(tf.shape(batch_tensor_2d)[0] * tf.shape(batch_tensor_2d)[1]),
                     tf.shape(batch_tensor_2d)[1]), -1)  # shape=[50,1], tensor [[55],[110],165....]

        print index_offset
        flattened_indices = batch_indices_2d + index_offset

        indices_flat = tf.reshape(flattened_indices, [-1])
        res = tf.gather(tensor_flat, flattened_indices)

        res_reshaped = tf.reshape(res, shape=tf.shape(batch_indices_2d))
        return res

    @staticmethod
    def gaussian_noise_layer(input_layer, std, name=None):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32, name=name)
        return input_layer + noise

    @staticmethod
    def tf_get_mask_from_seqlen_2D(tensor_2d_seqlen, name):
        """
        Generates mask from a 2d sequence length tensor
        :param tensor_2d_seqlen: Tensor with seqlens, 2d [batch_size, numbre_lens]
        :param name: Name of the output mask tensor
        :return: Mask for a 2d seq len vector.
        """
        input_candidates_in_story_pointer_seq_len_reshape = tf.reshape(tensor_2d_seqlen, [
            tf.reduce_prod(tf.shape(tensor_2d_seqlen))])

        input_candidates_in_story_pointer_seq_len_seq_mask = tf.cast(
            tf.sequence_mask(input_candidates_in_story_pointer_seq_len_reshape), dtype=tf.float32)
        shape = tf.shape(tensor_2d_seqlen)
        sinput_candidates_in_story_pointer_mask = tf.reshape(input_candidates_in_story_pointer_seq_len_seq_mask,
                                                                 [shape[0], shape[1], -1],
                                                                 name=name)
        return sinput_candidates_in_story_pointer_mask

    @staticmethod
    def tf_mask_inverse(mask):
        """
        Inversed padding mask: [1,1,1,0,0] -> [0,0,0,1,1]
        :param mask: Normal zero-padding mask [1,1,1,0,0]
        :return: Inversed mask:[0,0,0,1,1]
        """
        mask_inverse = tf.abs(mask - 1)

        return mask_inverse

    @staticmethod
    def tf_mask_to_softmaxmask(mask):
        """
        Convert zero padding mask to Softmax mask: mask [1, 1, 1, 0, 0] to [1, 1, 1, -1000000, -1000000] which used before a Softmax results in [0.33, 0.33, 0.33, 0, 0]
        :param mask: Mask tensor in the form [1, 1, 1, 0, 0]
        :return: Mask to be used before softmax for padded data [1, 1, 1, -1000000, -1000000] 
        """
        mask_softmax = mask + (mask - 1) * pow(2, 31)

        return mask_softmax