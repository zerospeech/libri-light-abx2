# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import random
from typing import Any, Callable

import numpy as np
import torch
from typing_extensions import LiteralString

from .models import Pooling


def normalize_with_singularity(x) -> torch.Tensor:
    r"""
    Normalize the given vector across the third dimension.
    Extend all vectors by eps=1e-12 to put the null vector at the maximal
    cosine distance from any non-null vector.
    """
    S, H = x.size()
    norm_x = (x ** 2).sum(dim=1, keepdim=True)

    x /= torch.sqrt(norm_x)
    zero_vals = (norm_x == 0).view(S)
    x[zero_vals] = 1 / math.sqrt(H)
    border_vect = torch.zeros((S, 1), dtype=x.dtype, device=x.device) + 1e-12
    border_vect[zero_vals] = -2 * 1e12
    return torch.cat([x, border_vect], dim=1)


def load_item_file(
        path_item_file: str,
) -> tuple[
    dict[str, list[list[Any]]], dict[str, int], dict[str, int], dict[str, int]
]:
    r"""Load a .item file indicating the triplets for the ABX score. The
    input file must have the following format:
    line 0 : whatever (not read)
    line > 0: #file_ID onset offset #phone prev-phone next-phone speaker
    onset : begining of the triplet (in s)
    onset : end of the triplet (in s)

    Returns a tuple of files_data, context_match, phone_match, speaker_match where
            files_data: dictionary whose key is the file id, and the value is the list of item tokens in that file, each item in turn
                            given as a list of onset, offset, context_id, phone_id, speaker_id.
            phone_match is a dictionary of the form { phone_str: phone_id }.
            context_match is a dictionary of the form { prev_phone_str+next_phone_str: context_id }.
            speaker_match is a dictionary of the form { speaker_str: speaker_id }.
            The id in each case is iterative (0, 1 ...)
    """
    with open(path_item_file, "r") as file:
        item_f_lines = file.readlines()[1:]

    item_f_lines = [x.replace("\n", "") for x in item_f_lines]

    # key: fileID, value: a list of items, each item in turn given as a list of
    # onset, offset, context_id, phone_id, speaker_id (see below for the id constructions)
    files_data: dict[str, list[list[Any]]] = {}

    # Provide a phone_id for each phoneme type (0, 1 ...)
    phone_match: dict[str, int] = {}
    context_match: dict[str, int] = {}  # ... context_id ...
    speaker_match: dict[str, int] = {}  # ... speaker_id ...

    for line in item_f_lines:
        items = line.split()
        assert len(items) == 7  # assumes 7-column files
        fileID = items[0]
        if fileID not in files_data:
            files_data[fileID] = []

        onset, offset = float(items[1]), float(items[2])
        phone = items[3]

        speaker = items[6]
        context = "+".join([items[4], items[5]])

        if phone not in phone_match:
            # We increment the id by 1 each time a new phoneme type is found
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

        files_data[fileID].append(
            [onset, offset, context_id, phone_id, speaker_id]
        )

    return files_data, context_match, phone_match, speaker_match


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
                    curr_group[i - 1].append(curr_group[i])
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
            curr_group[i - 1].append(curr_group[i])
        out_groups += curr_group[0]

    return in_index, out_groups


class ABXFeatureLoader:
    def __init__(
            self,
            pooling: Pooling,
            seed_n: int,
            path_item_file: str,
            seqList: list[tuple[str, LiteralString]],
            featureMaker: Callable,
            stepFeature: float,
            normalize: bool,
    ):
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

        random.seed(seed_n)
        (
            files_data,
            self.context_match,
            self.phone_match,
            self.speaker_match,
        ) = load_item_file(path_item_file)
        self.seqNorm = True
        self.stepFeature = stepFeature
        self.loadFromFileData(
            pooling, files_data, seqList, featureMaker, normalize
        )

    def pool(self, feature: torch.Tensor, pooling: Pooling) -> torch.Tensor:
        match pooling:
            case Pooling.NONE:
                return feature
            case Pooling.MEAN:
                # vector avg. But keep the original shape.
                # So e.g. if we had 4 frames with 51 feature dimensions [4,51],
                # we will get back [1,51], not [51]
                return feature.mean(dim=0, keepdim=True)
            case Pooling.HAMMING:
                h: np.ndarray = np.hamming(feature.size(0))
                np_f: np.ndarray = feature.detach().cpu().numpy()
                # weight vec dot feature matrix: each row/frame gets its own
                # hamming weight and all the rows are summed into a single
                # vector. Then divide by sum of weights. Finally, reshape
                # into original shape.
                pooled: np.ndarray = (h.dot(np_f) / sum(h))[None, :]
                return torch.from_numpy(pooled)
            case other:
                raise ValueError("Invalid value for pooling.")

    def start_end_indices(
            self,
            phone_start: Any,
            phone_end: Any,
            all_features: torch.Tensor,
            stepFeature: float,
    ) -> tuple[int, int]:
        index_start = max(0, int(math.ceil(stepFeature * phone_start - 0.5)))
        index_end = int(
            min(
                all_features.size(0),
                int(math.floor(stepFeature * phone_end - 0.5)),
            )
        )
        return index_start, index_end

    def append_feature(
            self,
            index_start,
            index_end,
            totSize: int,
            all_features: torch.Tensor,
            context_id: Any,
            phone_id: Any,
            speaker_id: Any,
            data: list[torch.Tensor],
            manifest: list[Any],
            pooling: Pooling,
    ) -> int:
        """Build and append the feature to the features data list.
        Add information on it to the manifest, i.e. to self.features.
        Return the total size i.e. the total number of frames added to the data thus far."""
        feature = all_features[index_start:index_end]
        feature = self.pool(feature, pooling)
        start_i = totSize
        loc_size = feature.shape[0]
        manifest.append([start_i, loc_size, context_id, phone_id, speaker_id])
        data.append(feature)
        return totSize + loc_size

    def loadFromFileData(
            self,
            pooling: Pooling,
            files_data: dict[str, list[list[Any]]],
            seqList: list[tuple[str, LiteralString]],
            feature_maker: Callable,
            normalize: bool,
    ):

        # self.features[i]: index_start, size, context_id, phone_id, speaker_id
        # This is like a manifest of what is in self.data[i]
        self.features = []
        self.INDEX_CONTEXT = 2
        self.INDEX_PHONE = 3
        self.INDEX_SPEAKER = 4
        # data[i] is the data for a given item. data is no longer discriminated by file
        data: list[torch.Tensor] = []

        totSize = 0

        print("Building the input features...")

        for index, vals in enumerate(seqList):

            fileID, file_path = vals
            if fileID not in files_data:
                # file in submission not in the item file
                continue

            all_features: torch.Tensor = feature_maker(file_path)
            if normalize:
                all_features = normalize_with_singularity(all_features)

            all_features = all_features.detach().cpu()

            # The list of item tokens in a file as defined by the item file
            # Each item is given as a list of onset, offset, context_id, phone_id, speaker_id
            phone_data = files_data[fileID]

            for (
                    phone_start,
                    phone_end,
                    context_id,
                    phone_id,
                    speaker_id,
            ) in phone_data:
                index_start, index_end = self.start_end_indices(
                    phone_start, phone_end, all_features, self.stepFeature
                )
                if (
                        index_start >= all_features.size(0)
                        or index_end <= index_start
                ):
                    continue
                totSize = self.append_feature(
                    index_start,
                    index_end,
                    totSize,
                    all_features,
                    context_id,
                    phone_id,
                    speaker_id,
                    data,
                    self.features,
                    pooling,
                )

        print("...done")
        self.data = torch.cat(data, dim=0)
        self.feature_dim = self.data.size(1)

    def get_data_device(self) -> torch.device:
        return self.data.device

    def cuda(self):
        self.data = self.data.cuda()

    def cpu(self):
        self.data = self.data.cpu()

    def get_max_group_size(self, i_group, i_sub_group):
        id_start, id_end = self.group_index[i_group][i_sub_group]
        return max([self.features[i][1] for i in range(id_start, id_end)])

    def get_ids(self, index) -> tuple[int, int, int]:
        context_id, phone_id, speaker_id = self.features[index][2:]
        return context_id, phone_id, speaker_id

    def __getitem__(self, index):
        i_data, out_size, context_id, phone_id, speaker_id = self.features[index]
        return (
            self.data[i_data: (i_data + out_size)],
            out_size,
            (context_id, phone_id, speaker_id),
        )

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

    def get_iterator(self, mode, max_size_group):
        if mode == "within":
            return ABXWithinGroupIterator(self, max_size_group)
        if mode == "across":
            return ABXAcrossGroupIterator(self, max_size_group)
        if mode == "any":
            raise ValueError(f"Mode not yet supported: {mode}")
        raise ValueError(f"Invalid mode: {mode}")


class ABXIterator:
    r"""
    Base class building ABX's triplets.
    """

    def __init__(self, abxDataset, max_size_group):
        self.max_size_group = max_size_group
        self.dataset = abxDataset
        self.len = 0

        self.index_csp, self.groups_csp = get_features_group(
            abxDataset.features,
            [
                abxDataset.INDEX_CONTEXT,
                abxDataset.INDEX_SPEAKER,
                abxDataset.INDEX_PHONE,
            ],
        )

    def get_group(self, i_start, i_end):
        data = []
        max_size = 0
        to_take = list(range(i_start, i_end))
        if i_end - i_start > self.max_size_group:
            to_take = random.sample(to_take, k=self.max_size_group)
        for i in to_take:
            loc_data, loc_size, loc_id = self.dataset[self.index_csp[i]]
            max_size = max(loc_size, max_size)
            data.append(loc_data)

        N = len(to_take)
        out_data = torch.zeros(
            N,
            max_size,
            self.dataset.feature_dim,
            device=self.dataset.get_data_device(),
        )
        out_size = torch.zeros(
            N, dtype=torch.long, device=self.dataset.get_data_device()
        )

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


class ABXWithinGroupIterator(ABXIterator):
    r"""
    Iterator giving the triplets for the ABX within score.
    """

    def __init__(self, abxDataset, max_size_group):

        super(ABXWithinGroupIterator, self).__init__(abxDataset, max_size_group)
        self.symmetric = True

        for context_group in self.groups_csp:  # always within context
            for speaker_group in context_group:
                if len(speaker_group) > 1:
                    for i_start, i_end in speaker_group:
                        if i_end - i_start > 1:
                            self.len += len(speaker_group) - 1

    def __iter__(self):
        for i_c, context_group in enumerate(self.groups_csp):
            for i_s, speaker_group in enumerate(context_group):
                n_phones = len(speaker_group)
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
                        data_b, size_b, id_b = self.get_group(i_start_b, i_end_b)
                        data_a, size_a, id_a = self.get_group(i_start_a, i_end_a)

                        out_coords = id_a[2], id_a[1], id_b[1], id_a[0]
                        yield out_coords, (data_a, size_a), (data_b, size_b), (
                            data_a,
                            size_a,
                        )

    def get_board_size(self):

        return (
            self.dataset.get_n_speakers(),
            self.dataset.get_n_phone(),
            self.dataset.get_n_phone(),
            self.dataset.get_n_context(),
        )


class ABXAcrossGroupIterator(ABXIterator):
    r"""
    Iterator giving the triplets for the ABX across score.
    """

    def __init__(self, abxDataset, max_size_group):

        super(ABXAcrossGroupIterator, self).__init__(abxDataset, max_size_group)
        self.symmetric = False
        self.get_speakers_from_cp = {}
        self.max_x = 5

        for context_group in self.groups_csp:
            for speaker_group in context_group:
                for i_start, i_end in speaker_group:
                    c_id, p_id, s_id = self.dataset.get_ids(
                        self.index_csp[i_start]
                    )
                    if c_id not in self.get_speakers_from_cp:
                        self.get_speakers_from_cp[c_id] = {}
                    if p_id not in self.get_speakers_from_cp[c_id]:
                        self.get_speakers_from_cp[c_id][p_id] = {}
                    self.get_speakers_from_cp[c_id][p_id][s_id] = (i_start, i_end)

        for context_group in self.groups_csp:
            for speaker_group in context_group:
                if len(speaker_group) > 1:
                    for i_start, i_end in speaker_group:
                        c_id, p_id, s_id = self.dataset.get_ids(
                            self.index_csp[i_start]
                        )
                        self.len += (len(speaker_group) - 1) * (
                            min(
                                self.max_x,
                                len(self.get_speakers_from_cp[c_id][p_id]) - 1,
                            )
                        )

    def get_other_speakers_in_group(self, i_start_group):
        c_id, p_id, s_id = self.dataset.get_ids(self.index_csp[i_start_group])
        return [
            v
            for k, v in self.get_speakers_from_cp[c_id][p_id].items()
            if k != s_id
        ]

    def get_abx_triplet(self, i_a, i_b, i_x):
        i_start_a, i_end_a = i_a
        data_a, size_a, id_a = self.get_group(i_start_a, i_end_a)

        i_start_b, i_end_b = i_b
        data_b, size_b, id_b = self.get_group(i_start_b, i_end_b)

        i_start_x, i_end_x = i_x
        data_x, size_x, id_x = self.get_group(i_start_x, i_end_x)

        out_coords = id_a[2], id_a[1], id_b[1], id_a[0], id_x[2]
        return out_coords, (data_a, size_a), (data_b, size_b), (data_x, size_x)

    def __iter__(self):
        for i_c, context_group in enumerate(self.groups_csp):
            for i_s, speaker_group in enumerate(context_group):
                n_phones = len(speaker_group)
                if n_phones == 1:
                    continue

                for i_a in range(n_phones):
                    i_start_a, i_end_a = self.groups_csp[i_c][i_s][i_a]
                    ref = self.get_other_speakers_in_group(i_start_a)
                    if len(ref) > self.max_x:
                        speakers_a = random.sample(ref, k=self.max_x)
                    else:
                        speakers_a = ref

                    for i_start_x, i_end_x in speakers_a:

                        for i_b in range(n_phones):
                            if i_b == i_a:
                                continue

                            i_start_b, i_end_b = self.groups_csp[i_c][i_s][i_b]
                            yield self.get_abx_triplet(
                                (i_start_a, i_end_a),
                                (i_start_b, i_end_b),
                                (i_start_x, i_end_x),
                            )

    def get_board_size(self):

        return (
            self.dataset.get_n_speakers(),
            self.dataset.get_n_phone(),
            self.dataset.get_n_phone(),
            self.dataset.get_n_context(),
            self.dataset.get_n_speakers(),
        )
