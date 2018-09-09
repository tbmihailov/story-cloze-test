import sys
import warnings


class LabelDictionary(dict):
    """This class implements a dictionary of labels. Labels as mapped to
    integers, and it is efficient to retrieve the label name from its
    integer representation, and vice-versa."""

    def __init__(self, label_names=[], start_index=0):
        self.names = {}
        # self.index = start_index
        self.start_index = start_index
        for name in label_names:
            self.add(name)

    def add(self, name):
        if name in self:
            warnings.warn('Ignoring duplicated label ' + name)
            return self[name]

        label_id = len(self.names) + self.start_index
        self[name] = label_id
        self.names[label_id] = name

        # self.index = label_id + 1
        return label_id

    def get_label_name(self, label_id):
        return self.names[label_id]

    def get_label_id(self, name):
        return self[name]

    def set_dict(self, name_idx_dict):
        for name, idx in name_idx_dict.iteritems():
            self[name] = idx
            self.names[idx] = name
