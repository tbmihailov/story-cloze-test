import codecs

def lowercase_list(lst, lowercase):
    """
    Converts list of string/unicode to lowercase
    :param lst:  List of strings/unicodes
    :param lowercase: Whether to convert or keep as is
    :return: List of string/unicode
    """
    if lowercase:
        return [x.lower() for x in lst]
    else:
        return lst

def transform_tokens_to_ids(tokens, vocab_dict, pad_id, unknown_word_id, lowercase):
    """
    Converts tokens to ids from the matches in the vocab
    :param tokens:
    :param vocab_dict:
    :param pad_id:
    :param unknown_word_id:
    :param lowercase:
    :return:
    """
    return [vocab_dict[x] if x in vocab_dict else unknown_word_id for x in
            lowercase_list(tokens, lowercase)]


def convert_raw_vocab_to_lowercase(vocab_raw):
    """
    Converts list of (word, freq) to list of lowercased word and the total frequency

    :param vocab_raw: List of (word, freq)
    :return: Lowercased words with frequencies
    """
    vocab_low = {}
    for word, freq in vocab_raw:
        word_low = word.lower()
        if word_low in vocab_low:
            vocab_low[word_low] += freq
        else:
            vocab_low[word_low] = freq

    vocab_transformed_lst = [(w, freq) for w, freq in vocab_low.iteritems()]
    sorted(vocab_transformed_lst, key=lambda a: a[1], reverse=True)

    return vocab_transformed_lst


def extract_char_vocab_from_word_vocab(word_vocab):
    """
    Extracts character vocab from a list of words. Use this if the word vocab is already
    :param word_vocab: List of words or tuple with first item word. Words should be case sensitive.
    :return: List of (char, freq) ex: ("char", 33) where 33 if the number total number of occurances of the char in the word vocab
    """
    if (word_vocab.__class__ != list):
        raise TypeError(
            "word_vocab should be list of string/unicode or list of tuples with 1st item word(string/unicode) ex: (\"apple\", 123)")

    if len(word_vocab) == 0:
        raise ValueError("word_vocab must be a list with at least 1 string item!")

    chars = {}
    for item in word_vocab:
        if item.__class__ == tuple:
            word = item[0]
        else:
            word = item

        for ch in word:
            if ch in chars:
                chars[ch] += 1
            else:
                chars[ch] = 1

    sort_by_col = 0  # aphlabet order
    chars_list = [(ch, freq) for ch, freq in chars.iteritems()]
    sorted(chars_list, key=lambda a: a[sort_by_col], reverse=(sort_by_col > 0))

    return chars_list


def save_vocab_to_text_file(vocab, text_file):
    """
    Saved vocab to file: (id, word, freq) to => "word\tfreq". First row is the number of words in the vocab
    :param vocab: List of (id, word, freq)
    :param text_file: Text file to save the vocab to
    :return: Items saved
    """

    num_words = 0

    f = codecs.open(text_file, mode="w", encoding="utf-8")
    f.write("%s\n" % len(vocab))
    for item in vocab:
        f.write("%s\t%s\n" % (item[1], item[2]))
        num_words += 1
    f.close()

    return num_words

class DataUtilities_Base(object):
    def evaluator(self):
        return None

    def labels_cnt(self):
        return 0

    def knowledge_settings(self, corpus_settings, knowledge_run_name):
        return None

    def settings(self, data_run_name_dir):
        return None