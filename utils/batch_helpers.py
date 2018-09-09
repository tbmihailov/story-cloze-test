import random
import numpy as np

def pad(seq, pad_value, to_size):
    pad_seq = []
    if len(seq) > to_size:
        pad_seq = seq[:to_size]
    else:
        pad_seq = seq[:] + [pad_value] * (to_size - len(seq))
    return pad_seq


def pad_data_and_return_seqlens(data, pad_value=0):
    batch_data_seqlens = np.asarray([len(a) for a in data])
    max_len = max(batch_data_seqlens)
    batch_data_padded_x = np.asarray([pad(a, pad_value, max_len) for a in data])
    return batch_data_padded_x, batch_data_seqlens

def pad_and_get_seq_len_and_mask_rank3(data, pad_value=0):
    # data is of rank 3: [batch_size, number_tokens, number_deps]

    # number tokens
    batch_data_seqlens = [len(a) for a in data]
    max_len_seq = max(batch_data_seqlens)

    # max number of deps
    max_sub_length = 0
    for item in data:
        # print "item:%s"%item
        for token in item:
            if len(token) > max_sub_length:
                max_sub_length = len(token)

    # Build the input
    data_padded = []
    mask = []
    data_padded_seqlens = []
    for item in data:
        item_padded = []
        item_mask = []
        item_seqlens = []
        for token_deps in item:

            token_deps_padded = token_deps + [pad_value]*(max_sub_length-len(token_deps))
            item_padded.append(token_deps_padded)

            token_deps_mask = [1]*len(token_deps)+[0]*(max_sub_length-len(token_deps))
            item_mask.append(token_deps_mask)
            item_seqlens.append(max(1, sum(token_deps_mask)))

        item_padded = item_padded + (max_len_seq-len(item_padded))*[[pad_value]*max_sub_length]
        data_padded.append(item_padded)

        item_mask = item_mask + (max_len_seq - len(item_mask)) * [[pad_value] * max_sub_length]
        mask.append(item_mask)

        item_seqlens = item_seqlens + (max_len_seq - len(item_mask)) * [1]
        data_padded_seqlens.append(item_seqlens)

        data_padded_seqlens_padded, _ = pad_data_and_return_seqlens(data_padded_seqlens)
    return np.asarray(data_padded), np.asarray(data_padded_seqlens_padded), np.asarray(mask)


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a data_utils. Shuffles data and then batch
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = data_size / batch_size + (0 if data_size % batch_size == 0 else 1)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



def batch_iter_by_batch_shuffle(data, batch_size, num_epochs, shuffle=False):
    """
    Makes batches and then suffle.
    """

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = data_size / batch_size + (0 if data_size % batch_size == 0 else 1)
    for epoch in range(num_epochs):
        batches = []

        # Shuffle the data at each epoch
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            batches.append(data[start_index:end_index])

        batches_size = len(batches)

        batch_indeces = np.arange(batches_size)

        if shuffle:
            batch_indeces = np.random.permutation(batch_indeces)

        for batch_id in batch_indeces:
            yield batches[batch_id]

def batch_iter_random_batch_per_steps(data, batch_sizes, num_steps, random_seed=42):
    """
    Generates a batch iterator for a data_utils.
    """
    data = np.array(data)

    for step in range(num_steps):
        # Shuffle the data at each epoch
        curr_batch_size = random.choice(batch_sizes)
        batch_data = random.sample(data, curr_batch_size)
        yield batch_data

def batch_iter_plain(data, data_size, batch_size, num_epochs, num_steps):
    """
    Iterates over the data and returns batches

    :param data: The input data
    :param data: The size of the data - in case that the input data is reader 
    :param batch_size: The size of the batches to return
    :param num_epochs: Number of epochs to iterate (if num_steps==0)
    :param num_steps: Number of steps to iterate
    :return: Iterator for the given data
    """

    num_batches_per_epoch = data_size / batch_size + (0 if data_size % batch_size == 0 else 1)
    curr_item_idx = 0


    max_num_steps = num_epochs * data_size if num_steps == 0 else num_steps

    if num_steps > 0:
        num_epochs = data_size / data_size

    data_batch = []
    steps_cnt = 0
    for epoch in range(num_epochs):
        for item in data:
            data_batch.append(item)
            curr_item_idx += 1

            if steps_cnt > max_num_steps:
                break

            if (curr_item_idx) % batch_size == 0:
                yield data_batch
                data_batch = []
                steps_cnt += 1
                continue

    if len(data_batch) > 0:
        yield data_batch

def get_flat_values_for_candidate_pointers(batch_candidates_in_story_pointer):
    """
    Converts candidate-wise pointers in the full-story to a list of memories in the story
     and returns a new per-candidate pointers in the memory. 
    :param batch_candidates_in_story_pointer: 
    :return: flat_pointers, per-candidate memory pointers in the flat_pointers 
    """
    batch_story_memory_pointers = []
    batch_story_memory_pointers_cand_point = []
    for cand_points in batch_candidates_in_story_pointer:
        memory_pointer = []
        memory_pointer_cand_pointer = []
        for cand_point in cand_points:
            curr_pnts = []
            for pnt in cand_point:
                curr_pnts.append(len(memory_pointer))
                memory_pointer.append(pnt)
            memory_pointer_cand_pointer.append(curr_pnts)

        batch_story_memory_pointers_cand_point.append(memory_pointer_cand_pointer)
        batch_story_memory_pointers.append(memory_pointer)

    return batch_story_memory_pointers, batch_story_memory_pointers_cand_point

######################################
########Tests #########
######################################

def test_cases_batch_iter_plain():
    """
    Test batch_iter_plain(data, data_size, batch_size, num_epochs, num_steps):
    :return: 
    """
    def gen_data_iter_int(size):
        for i in range(size):
            yield i

    def test_batch_iter_even_items_size():
        """
        Checks if batching is proper for even number of baches and size
        :return: 
        """

        print "Checks if batching is proper for even number of baches and size:"
        batch_size = 30
        num_epochs = 3
        num_steps = 0
        data_size = 90

        data = gen_data_iter_int(data_size)

        batch_iter = batch_iter_plain(data, data_size, batch_size, num_epochs, num_steps)

        items_cnt = 0
        batch_idx = 0
        for batch in batch_iter:
            items_cnt += len(batch)
            print "item %s: %s" % (batch_idx, str(batch))

        assert items_cnt == data_size

        print "test_batch_iter_even_items_size pass"

    def test_batch_iter_uneven_items_size():
        """
        Checks if batching is proper for even number of baches and size
        :return: 
        """

        print "Checks if batching is proper for even number of baches and size:"
        batch_size = 30
        num_epochs = 3
        num_steps = 0
        data_size = 97

        data = gen_data_iter_int(data_size)

        batch_iter = batch_iter_plain(data, data_size, batch_size, num_epochs, num_steps)

        items_cnt = 0
        batch_idx = 0
        for batch in batch_iter:
            items_cnt += len(batch)
            print "item %s: %s" % (batch_idx, str(batch))

        assert items_cnt == data_size

        print "test_batch_iter_even_items_size pass"

    test_batch_iter_even_items_size()
    # test_batch_iter_even_items_size()

if __name__ == '__main__':
    print "Batch helpers"
    test_cases_batch_iter_plain()
