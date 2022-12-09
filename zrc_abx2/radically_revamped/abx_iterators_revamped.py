# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# This file should supercede abx_iterators.py in ABX_src folder
# and be used in conjunction with a modified eval_ABX.py version 
# to account for the differing Iterator class names

import torch
import math
import random

# these indices used to be within FeatureLoader, but I see no reason for this

INDEX_CONTEXT = 2
INDEX_PHONE = 3
INDEX_SPEAKER = 4

def normalize_with_singularity(x):
    r"""
    Normalize the given vector across the third dimension.
    Extend all vectors by eps=1e-12 to put the null vector at the maximal
    cosine distance from any non-null vector.
    """
    S, H = x.size()
    norm_x = (x**2).sum(dim=1, keepdim=True)

    x /= torch.sqrt(norm_x)
    zero_vals = (norm_x == 0).view(S)
    x[zero_vals] = 1 / math.sqrt(H)
    border_vect = torch.zeros((S, 1),
                              dtype=x.dtype,
                              device=x.device) + 1e-12
    border_vect[zero_vals] = -2*1e12
    return torch.cat([x, border_vect], dim=1)


def load_item_file(path_item_file):
    r""" Load a .item file indicating the triplets for the ABX score. The
    input file must have the following fomat:
    line 0 : whatever (not read)
    line > 0: #file_ID onset offset #phone prev-phone next-phone speaker
    onset : begining of the triplet (in s)
    offset : end of the triplet (in s)
    """
    with open(path_item_file, 'r') as file:
        data = file.readlines()[1:]

    data = [x.replace('\n', '') for x in data]

    out = {}

    phone_match = {}
    speaker_match = {}
    context_match = {}

    for line in data:
        items = line.split()
        assert(len(items) == 7)  # assumes 7-column files
        fileID = items[0]
        if fileID not in out:
            out[fileID] = []

        onset, offset = float(items[1]), float(items[2])
        phone = items[3]
        
        speaker = items[6]
        context = '+'.join([items[4], items[5]])

        if phone not in phone_match:
            s = len(phone_match)
            phone_match[phone] = s
        phone_id = phone_match[phone]
        
        if context not in context_match:
            s = len(context_match)
            context_match[context] = s
        context_id = context_match[context]

        if speaker not in speaker_match:
            s = len(speaker_match)
            speaker_match[speaker] = s
        speaker_id = speaker_match[speaker]

        out[fileID].append([onset, offset, context_id, phone_id, speaker_id])

    return out, context_match, phone_match, speaker_match


def get_features_group(in_data, index_order):

    in_index = list(range(len(in_data)))
    in_index.sort(key=lambda x: [in_data[x][i] for i in index_order])
    out_groups = []
    last_values = [in_data[in_index[0]][i] for i in index_order]
    i_s = 0
    curr_group = [[] for i in index_order]
    n_orders = len(index_order) - 1
    tmp = [in_data[i] for i in in_index]

    for index, item in enumerate(tmp):
        for order_index, order in enumerate(index_order):
            if item[order] != last_values[order_index]:
                curr_group[-1].append((i_s, index))
                for i in range(n_orders, order_index, -1):
                    curr_group[i-1].append(curr_group[i])
                    curr_group[i] = []
                if order_index == 0:
                    out_groups += curr_group[0]
                    curr_group[0] = []
                last_values = [item[i] for i in index_order]
                i_s = index
                break

    if i_s < len(in_data):
        curr_group[-1].append((i_s, len(in_data)))
        for i in range(n_orders, 0, -1):
            curr_group[i-1].append(curr_group[i])
        out_groups += curr_group[0]

    return in_index, out_groups


class ABXFeatureLoader:

    def __init__(self,
                 path_item_file,
                 seqList,
                 featureMaker,
                 stepFeature,
                 normalize):
        """
        Args:
            path_item_file (str): path to the .item files containing the ABX
                                  triplets
            seqList (list): list of items (fileID, path) where fileID refers to
                            the file's ID as used in path_item_file, and path
                            is the actual path to the input audio sequence
            featureMaker (function): either a function or a callable object.
                                     Takes a path as input and outputs the
                                     feature sequence corresponding to the
                                     given file.
            normalize (bool): if True all input features will be noramlized
                              across the channels dimension.

        Note:
        You can use this dataset with pre-computed features. For example, if
        you have a collection of features files in the torch .pt format then
        you can just set featureMaker = torch.load.
        """

        files_data, self.context_match, self.phone_match, self.speaker_match = \
            load_item_file(path_item_file)
        self.seqNorm = True
        self.stepFeature = stepFeature
        self.loadFromFileData(files_data, seqList, featureMaker, normalize)

    def loadFromFileData(self, files_data, seqList, feature_maker, normalize):

        # self.features[i]: index_start, size, context_id, phone_id, speaker_id
        self.features = []
        # indices need not be stored here specifically, can be used elsewhere, surely
        
        data = []

        totSize = 0

        print("Building the input features...")

        for index, vals in enumerate(seqList):

            fileID, file_path = vals
            if fileID not in files_data:
                continue

            features = feature_maker(file_path)
            if normalize:
                features = normalize_with_singularity(features)

            features = features.detach().cpu()

            phone_data = files_data[fileID]

            for phone_start, phone_end, context_id, phone_id, speaker_id in phone_data:

                index_start = max(
                    0, int(math.ceil(self.stepFeature * phone_start - 0.5)))
                index_end = min(features.size(0),
                                int(math.floor(self.stepFeature * phone_end - 0.5)))

                if index_start >= features.size(0) or index_end <= index_start:
                    continue

                loc_size = index_end - index_start
                self.features.append([totSize, loc_size, context_id,
                                      phone_id, speaker_id])
                data.append(features[index_start:index_end])
                totSize += loc_size

        print("...done")

        self.data = torch.cat(data, dim=0)
        self.feature_dim = self.data.size(1)

    def get_data_device(self):
        return self.data.device

    def cuda(self):
        self.data = self.data.cuda()

    def cpu(self):
        self.data = self.data.cpu()

    def get_max_group_size(self, i_group, i_sub_group):
        id_start, id_end = self.group_index[i_group][i_sub_group]
        return max([self.features[i][1] for i in range(id_start, id_end)])

    def get_ids(self, index):
        context_id, phone_id, speaker_id = self.features[index][2:]
        return context_id, phone_id, speaker_id

    def __getitem__(self, index):
        i_data, out_size, context_id, phone_id, speaker_id = self.features[index]
        return self.data[i_data:(i_data + out_size)], out_size, (context_id, phone_id, speaker_id)

    def __len__(self):
        return len(self.features)

    def get_n_speakers(self):
        return len(self.speaker_match)

    def get_n_context(self):
        return len(self.context_match)

    def get_n_phone(self):
        return len(self.phone_match)

    def get_n_groups(self):
        return len(self.group_index)

    def get_n_sub_group(self, index_sub_group):
        return len(self.group_index[index_sub_group])

    def get_iterator(self, contextmode, speakermode, max_size_group):
        if contextmode == speakermode == 'within':
            return ABXWithinWithinGroupIterator(self, context, speaker, max_size_group)
        if (contextmode == 'across' and speakermode == "within") or
                (speakermode == "across" and contextmode == "within"):
            return ABXWithinAcrossGroupIterator(self, context, speaker, max_size_group)
        if ((contextmode == "any") ^ (speakermode == "any")):
            # if one and only one of the modes is "any"
            # then the other must be "within" or "across"
            if speakermode == "within" or contextmode == "within":
                return ABXAnyWithinGroupIterator(self, context, speaker, max_size_group)
            elif speakermode == "across" or contextmode == "across":
                return ABXAnyAcrossGroupIterator(self, context, speaker, max_size_group)
            # if neither one is, then the "any-any" combination is not supported:
        raise ValueError(f"Invalid mode: {mode}")


class ABXIterator:
    r"""
    Base class building ABX's triplets.
    """

    def __init__(self, abxDataset, contextmode, speakermode, max_size_group):
        self.max_size_group = max_size_group
        self.dataset = abxDataset
        self.len = 0
        
        self.index_order = self.get_mode_info(contextmode, speakermode)

        self.index_csp, self.groups_csp = \
            get_features_group(abxDataset.features,
                               self.index_order)
    
    def get_mode_info(contextmode, speakermode):
        # indices' order in index_order list determines 
        # the "within" & "across" variables
        # which are pre-determined within get_features_group
        # before we ever get to the within/across iterators
        # ("any" is handled by not grouping by that variable at all)
        if speakermode in ["within", "across"]:
            if contextmode == "within":
                return [INDEX_CONTEXT, INDEX_SPEAKER, INDEX_PHONE]
            elif contextmode == "across":
                if speakermode == "across":
                    raise ValueError(f"Mode not yet supported: across context, across speaker")
                return [INDEX_SPEAKER, INDEX_CONTEXT, INDEX_PHONE]
            elif contextmode == "any":
                return [INDEX_SPEAKER, INDEX_PHONE]
        elif speakermode == "any":
            if contextmode == "any":
                raise ValueError("Mode not yet supported: any context, any speaker")
            return [INDEX_CONTEXT, INDEX_PHONE]
        
    def get_group(self, i_start, i_end):
        data = []
        max_size = 0
        to_take = list(range(i_start, i_end))
        if i_end - i_start > self.max_size_group:
            random.seed(42)
            to_take = random.sample(to_take, k=self.max_size_group)
        for i in to_take:
            loc_data, loc_size, loc_id = self.dataset[self.index_csp[i]]
            max_size = max(loc_size, max_size)
            data.append(loc_data)

        N = len(to_take)
        out_data = torch.zeros(N, max_size,
                               self.dataset.feature_dim,
                               device=self.dataset.get_data_device())
        out_size = torch.zeros(N, dtype=torch.long,
                               device=self.dataset.get_data_device())

        for i in range(N):
            size = data[i].size(0)
            out_data[i, :size] = data[i]
            out_size[i] = size

        return out_data, out_size, loc_id

    def __len__(self):
        return self.len

    def get_board_size(self):
        r"""
        Get the output dimension of the triplet's space.
        """
        pass


class ABXWithinWithinGroupIterator(ABXIterator):
    r"""
    Iterator giving the triplets for the ABX within context, within speaker score.
    """

    def __init__(self, abxDataset, contextmode, speakermode, max_size_group):

        super(ABXWithinWithinGroupIterator, self).__init__(abxDataset, contextmode, speakermode,
                                                     max_size_group)
        self.symmetric = True

        # the ABXIterator object does the within/across disambiguation internally
        # when it puts together the self.groups_csp list-of-lists
        # so nothing needs done here to keep track of the within vs across variables:
        # we can just treat groups_csp neutrally
        
        for within_group in self.groups_csp: # always within context
            for within2_group in within_group:
                if len(within2_group) > 1:
                    for i_start, i_end in within2_group:
                        if i_end - i_start > 1:
                            self.len += (len(within2_group) - 1)

    def __iter__(self):
        for i_c, within_group in enumerate(self.groups_csp):
            for i_s, within2_group in enumerate(within_group):
                n_phones = len(within2_group)
                if n_phones == 1:
                    continue

                for i_a in range(n_phones):
                    i_start_a, i_end_a = self.groups_csp[i_c][i_s][i_a]
                    if i_end_a - i_start_a == 1:
                        continue

                    for i_b in range(n_phones):
                        if i_b == i_a:
                            continue

                        i_start_b, i_end_b = self.groups_csp[i_c][i_s][i_b]
                        data_b, size_b, id_b = self.get_group(i_start_b,
                                                              i_end_b)
                        data_a, size_a, id_a = self.get_group(i_start_a,
                                                              i_end_a)

                        out_coords = id_a[2], id_a[1], id_b[1], id_a[0]
                        yield out_coords, (data_a, size_a), (data_b, size_b), \
                            (data_a, size_a)

    def get_board_size(self):

        return (self.dataset.get_n_speakers(),
                self.dataset.get_n_phone(),
                self.dataset.get_n_phone(),
                self.dataset.get_n_context())


class ABXWithinAcrossGroupIterator(ABXIterator):
    r"""
    Iterator giving the ABX triplets for any form of
    "within" + "across" speakermode / contextmode combo
    """

    def __init__(self, abxDataset, contextmode, speakermode, max_size_group):

        super(ABXWithinAcrossGroupIterator, self).__init__(abxDataset, contextmode, speakermode,
                                                     max_size_group)
        
        self.symmetric = False
        self.get_values_from_group = {}
        self.max_x = 5
        
        # the ABXIterator object does the within/across disambiguation internally
        # when it puts together the self.groups_csp list-of-lists
        # so nothing needs done here to keep track of the within vs across variables:
        # we can just treat groups_csp neutrally
        
        for within_group in self.groups_csp:
            for across_group in within_group:
                for i_start, i_end in across_group:
                    c_id, p_id, s_id = self.dataset.get_ids(
                        self.index_csp[i_start])
                    if c_id not in self.get_values_from_group:
                        self.get_values_from_group[c_id] = {}
                    if p_id not in self.get_values_from_group[c_id]:
                        self.get_values_from_group[c_id][p_id] = {}
                    self.get_values_from_group[c_id][p_id][s_id] = (
                        i_start, i_end)

        for within_group in self.groups_csp:
            for across_group in within_group:
                if len(across_group) > 1:
                    for i_start, i_end in across_group:
                        c_id, p_id, s_id = self.dataset.get_ids(
                            self.index_csp[i_start])
                        self.len += (len(across_group) - 1) * (min(self.max_x,
                                                                    len(self.get_values_from_group[c_id][p_id]) - 1))

    def get_other_values_in_group(self, i_start_group):
        c_id, p_id, s_id = self.dataset.get_ids(self.index_csp[i_start_group])
        return [v for k, v in self.get_values_from_group[c_id][p_id].items() if k != s_id]

    def get_abx_triplet(self, i_a, i_b, i_x):
        i_start_a, i_end_a = i_a
        data_a, size_a, id_a = self.get_group(i_start_a, i_end_a)

        i_start_b, i_end_b = i_b
        data_b, size_b, id_b = self.get_group(i_start_b, i_end_b)

        i_start_x, i_end_x = i_x
        data_x, size_x, id_x = self.get_group(i_start_x, i_end_x)

        out_coords = id_a[2], id_a[1], id_b[1], id_a[0], id_x[2]
        return out_coords, (data_a, size_a), (data_b, size_b), \
            (data_x, size_x)

    def __iter__(self):
        for i_c, within_group in enumerate(self.groups_csp):
            for i_s, across_group in enumerate(within_group):
                n_phones = len(across_group)
                if n_phones == 1:
                    continue

                for i_a in range(n_phones):
                    i_start_a, i_end_a = self.groups_csp[i_c][i_s][i_a]
                    ref = self.get_other_values_in_group(i_start_a)
                    if len(ref) > self.max_x:
                        across_a = random.sample(ref, k=self.max_x)
                    else:
                        across_a = ref

                    for i_start_x, i_end_x in across_a:

                        for i_b in range(n_phones):
                            if i_b == i_a:
                                continue

                            i_start_b, i_end_b = self.groups_csp[i_c][i_s][i_b]
                            yield self.get_abx_triplet((i_start_a, i_end_a), (i_start_b, i_end_b), (i_start_x, i_end_x))

    def get_board_size(self):

        return (self.dataset.get_n_speakers(),
                self.dataset.get_n_phone(),
                self.dataset.get_n_phone(),
                self.dataset.get_n_context(),
                self.dataset.get_n_speakers())

class ABXAnyWithinGroupIterator(ABXIterator):
    r"""
    Iterator giving the ABX triplets for any form of
    "any" + "within" speakermode / contextmode combo
    """

    def __init__(self, abxDataset, contextmode, speakermode, max_size_group):

        super(ABXAnyWithinGroupIterator, self).__init__(abxDataset, contextmode, speakermode,
                                                     max_size_group)
        self.symmetric = True

        for within_group in self.groups_csp:
            if len(within_group) > 1:
                for i_start, i_end in within_group:
                    if i_end - i_start > 1:
                        self.len += (len(within_group) - 1)

    def __iter__(self):
        for i_s, within_group in enumerate(self.groups_csp):
            n_phones = len(within_group)
            if n_phones == 1:
                continue

            for i_a in range(n_phones):
                i_start_a, i_end_a = self.groups_csp[i_s][i_a]
                if i_end_a - i_start_a == 1:
                    continue

                for i_b in range(n_phones):
                    if i_b == i_a:
                        continue

                    i_start_b, i_end_b = self.groups_csp[i_s][i_b]
                    data_b, size_b, id_b = self.get_group(i_start_b,
                                                            i_end_b)
                    data_a, size_a, id_a = self.get_group(i_start_a,
                                                            i_end_a)

                    out_coords = id_a[1], id_a[0], id_b[0]
                    yield out_coords, (data_a, size_a), (data_b, size_b), \
                        (data_a, size_a)

    def get_board_size(self):

        return (self.dataset.get_n_speakers(),
                self.dataset.get_n_phone(),
                self.dataset.get_n_phone(),
                self.dataset.get_n_context(),
                self.dataset.get_n_speakers())


class ABXAnyAcrossGroupIterator(ABXIterator):
    r"""
    Iterator giving the ABX triplets for any form of
    "any" + "across" contextmode / speakermode combo
    """

    def __init__(self, abxDataset, contextmode, speakermode, max_size_group):

        super(ABXAnyAcrossGroupIterator, self).__init__(abxDataset, contextmode, speakermode,
                                                     max_size_group)
        self.symmetric = False
        self.get_values_from_group = {}
        self.max_x = 5

        for across_group in self.groups_csp:
            for i_start, i_end in across_group:
                p_id, s_id = self.dataset.get_ids(
                    self.index_sp[i_start])
                if p_id not in self.get_values_from_group:
                    self.get_values_from_group[p_id] = {}
                self.get_values_from_group[p_id][s_id] = (
                    i_start, i_end)

        for across_group in self.groups_csp:
            if len(across_group) > 1:
                for i_start, i_end in across_group:
                    p_id, s_id = self.dataset.get_ids(
                        self.index_sp[i_start])
                    self.len += (len(across_group) - 1) * (min(self.max_x,
                                                                len(self.get_values_from_group[p_id]) - 1))

    def get_other_values_in_group(self, i_start_group):
        p_id, s_id = self.dataset.get_ids(self.index_sp[i_start_group])
        return [v for k, v in self.get_values_from_group[p_id].items() if k != s_id]

    def get_abx_triplet(self, i_a, i_b, i_x):
        i_start_a, i_end_a = i_a
        data_a, size_a, id_a = self.get_group(i_start_a, i_end_a)

        i_start_b, i_end_b = i_b
        data_b, size_b, id_b = self.get_group(i_start_b, i_end_b)

        i_start_x, i_end_x = i_x
        data_x, size_x, id_x = self.get_group(i_start_x, i_end_x)

        out_coords = id_a[1], id_a[0], id_b[0], id_x[1]
        return out_coords, (data_a, size_a), (data_b, size_b), \
            (data_x, size_x)

    def __iter__(self):
        for i_s, across_group in enumerate(self.groups_csp):
            n_phones = len(across_group)
            if n_phones == 1:
                continue

            for i_a in range(n_phones):
                i_start_a, i_end_a = self.groups_csp[i_s][i_a]
                ref = self.get_other_values_in_group(i_start_a)
                if len(ref) > self.max_x:
                    across_a = random.sample(ref, k=self.max_x)
                else:
                    across_a = ref

                for i_start_x, i_end_x in across_a:

                    for i_b in range(n_phones):
                        if i_b == i_a:
                            continue

                        i_start_b, i_end_b = self.groups_csp[i_s][i_b]
                        yield self.get_abx_triplet((i_start_a, i_end_a), (i_start_b, i_end_b), (i_start_x, i_end_x))

    def get_board_size(self):

        return (self.dataset.get_n_speakers(),
                self.dataset.get_n_phone(),
                self.dataset.get_n_phone(),
                self.dataset.get_n_context(),
                self.dataset.get_n_speakers())
    
class ABXAcrossAcrossGroupIterator(ABXIterator):
    """
    """
    
class ABXAnyAnyGroupIterator(ABXIterator):
    """
    """