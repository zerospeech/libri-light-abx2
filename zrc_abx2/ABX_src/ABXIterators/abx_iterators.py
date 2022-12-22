# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# This is a modification of LibriLight ABX evaluation's abx_iterations.py.
#
# The original ABX takes the middle phone, its context (prev & next phones),
# and the speaker. It can run across- and within-speaker ABX, "within-context".


import random
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Union

import torch

from ..ABXDataset.abx_feature_dataset import ABXFeatureDataset
from ..ABXDataset.abx_item_file_loader import *
from ..models import *

GROUP_PRIORITY_INDEX = 0


def get_features_group(
    in_data: List[ManifestFeatureItem], index_order: List[int]
):
    """Returns: tuple[in_index, out_groups] where
    in_index: list[int]. Contains indices for reordering a list of
    ManifestFeatureItems according to the priority given in index_order.
    E.g. ANY-CONTEXT: reorder by speaker, then phoneme; WITHIN-CONTEXT: reorder
    by context, speaker, phoneme.
    out_groups: list[Any]. Takes the indices in in_index and divides them in
    groups into an n-dimensional matrix with the priority defined in index_order.
    E.g. if index_order is [CONTEXT_IDX, SPEAKER_IDX, PHONE_IDX], then each
    outermost 'row' of out_groups will delilmit a context, and each element in
    that row will delimit a single speaker group. Finally, at the innermost level,
    a tuple will mark the beginning and end indices for manifest items with the
    same phoneme in that context and speaker group.
    """

    in_index = list(range(len(in_data)))

    # For instance, if index_order = [SPEAKER_IDX, PHONE_IDX]
    # we get all the indexes for items from the first speaker first,
    # ordered by phoneme; then all the indexes for items of the second speaker
    # again ordered by phoneme, etc.
    in_index.sort(key=lambda x: [in_data[x][i] for i in index_order])
    out_groups = []
    # E.g. might be [0, 0] in the any-context condition for the speaker_id,
    # phoneme_id of the first item in the rearranged order
    last_values = [in_data[in_index[0]][i] for i in index_order]
    i_start = 0
    curr_group = [[] for i in index_order]
    n_orders = len(index_order) - 1
    tmp = [in_data[i] for i in in_index]
    for i_end, item in enumerate(tmp):
        for order_index, order in enumerate(index_order):
            if item[order] != last_values[order_index]:
                curr_group[-1].append((i_start, i_end))

                # This will run if there is a transition in one of the not-innermost
                # (rightmost) levels. I.e if
                # index_order=[CONTEXT_IDX, SPEAKER_IDX, PHONE_IDX], it will run if
                # there is a transition context or speaker.
                for i in range(n_orders, order_index, -1):
                    curr_group[i - 1].append(curr_group[i])
                    curr_group[i] = []

                # reset curr_group when the outermost group changes
                if order_index == GROUP_PRIORITY_INDEX:
                    out_groups += curr_group[0]
                    curr_group[0] = []

                last_values = [item[i] for i in index_order]
                i_start = i_end
                break

    if i_start < len(in_data):
        curr_group[-1].append((i_start, len(in_data)))
        for i in range(n_orders, 0, -1):
            curr_group[i - 1].append(curr_group[i])
        out_groups += curr_group[0]
    return in_index, out_groups


@dataclass
class ABXIterator:
    r"""
    Base class building ABX's triplets.
    """

    abxDataset: ABXFeatureDataset
    max_size_group: Any  # TODO: Type
    symmetric: bool
    reorder_priority: List[int]
    context_type: ContextType
    seed_n: int
    len: int = 0

    indices_items: List[int] = field(init=False)
    # WITHIN CONTEXT CONDITION:
    # context groups containing speaker groups
    # WITHOUT CONTEXT CONDITION:
    # speaker groups
    indices_item_groups: List[Any] = field(init=False)

    def __post_init__(self):
        random.seed(self.seed_n)
        self.indices_items, self.indices_item_groups = get_features_group(
            self.abxDataset.features_manifest, self.reorder_priority
        )

    def get_group(
        self, i_start, i_end
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        data = []
        max_size = 0
        to_take = list(range(i_start, i_end))
        if i_end - i_start > self.max_size_group:
            to_take = random.sample(to_take, k=self.max_size_group)
        loc_id: Tuple[int, ...] = ()
        for i in to_take:
            data_item = self.abxDataset[self.indices_items[i]]
            loc_data = data_item.data
            loc_size = data_item.out_size
            loc_id = self._get_loc_id(data_item, self.context_type)
            max_size = max(loc_size, max_size)
            data.append(loc_data)

        N = len(to_take)
        out_data = torch.zeros(
            N,
            max_size,
            self.abxDataset.feature_dim,
            device=self.abxDataset.get_data_device(),
        )
        out_size = torch.zeros(
            N, dtype=torch.long, device=self.abxDataset.get_data_device()
        )

        for i in range(N):
            size = data[i].size(0)
            out_data[i, :size] = data[i]
            out_size[i] = size

        return out_data, out_size, loc_id

    def _get_loc_id(
        self, data_item: ABXFeaturesDataItem, context_type: ContextType
    ) -> Union[Tuple[int, int, int], Tuple[int, int]]:
        if context_type == ContextType.WITHIN:
            return (
                data_item.context_id,
                data_item.phone_id,
                data_item.speaker_id,
            )
        elif context_type == ContextType.ANY:
            return (
                data_item.phone_id,
                data_item.speaker_id,
            )
        else:
            raise ValueError("Invalid context type.")

    def __len__(self):
        return self.len

    def get_board_size(self):
        r"""
        Get the output dimension of the triplet's space.
        """
        pass


class ABXWithinGroupIterator(ABXIterator):
    r"""
    Iterator giving the triplets for the ABX within speaker score.
    """

    def __init__(
        self,
        abxDataset: ABXFeatureDataset,
        max_size_group,
        reorder_priority: List[int],
        context_type: ContextType,
        seed_n: int,
    ):

        super().__init__(
            abxDataset=abxDataset,
            max_size_group=max_size_group,
            symmetric=True,
            reorder_priority=reorder_priority,
            context_type=context_type,
            seed_n=seed_n,
        )

        for context_group in self.indices_item_groups:  # always within context
            for speaker_group in context_group:
                if len(speaker_group) > 1:
                    for i_start, i_end in speaker_group:
                        if i_end - i_start > 1:
                            self.len += len(speaker_group) - 1

    def __iter__(self):
        for i_c, context_group in enumerate(self.indices_item_groups):
            for i_s, speaker_group in enumerate(context_group):
                n_phones = len(speaker_group)
                if n_phones == 1:
                    continue

                for i_a in range(n_phones):
                    i_start_a, i_end_a = self.indices_item_groups[i_c][i_s][i_a]
                    if i_end_a - i_start_a == 1:
                        continue

                    for i_b in range(n_phones):
                        if i_b == i_a:
                            continue

                        i_start_b, i_end_b = self.indices_item_groups[i_c][i_s][
                            i_b
                        ]
                        data_b, size_b, id_b = self.get_group(i_start_b, i_end_b)
                        data_a, size_a, id_a = self.get_group(i_start_a, i_end_a)

                        out_coords = id_a[2], id_a[1], id_b[1], id_a[0]
                        yield out_coords, (data_a, size_a), (data_b, size_b), (
                            data_a,
                            size_a,
                        )

    def get_board_size(self):

        return (
            self.abxDataset.get_n_speakers(),
            self.abxDataset.get_n_phone(),
            self.abxDataset.get_n_phone(),
            self.abxDataset.get_n_context(),
        )


class ABXAcrossGroupIterator(ABXIterator):
    r"""
    Iterator giving the triplets for the ABX across score.
    """

    def __init__(
        self,
        abxDataset: ABXFeatureDataset,
        max_size_group,
        reorder_priority: List[int],
        context_type: ContextType,
        seed_n: int,
    ):

        super().__init__(
            abxDataset=abxDataset,
            max_size_group=max_size_group,
            symmetric=False,
            reorder_priority=reorder_priority,
            context_type=context_type,
            seed_n=seed_n,
        )
        self.get_speakers_from_cp = {}
        self.max_x = 5

        for context_group in self.indices_item_groups:
            for speaker_group in context_group:
                for i_start, i_end in speaker_group:
                    c_id, p_id, s_id = self.abxDataset.get_ids(
                        self.indices_items[i_start]
                    )
                    if c_id not in self.get_speakers_from_cp:
                        self.get_speakers_from_cp[c_id] = {}
                    if p_id not in self.get_speakers_from_cp[c_id]:
                        self.get_speakers_from_cp[c_id][p_id] = {}
                    self.get_speakers_from_cp[c_id][p_id][s_id] = (i_start, i_end)

        for context_group in self.indices_item_groups:
            for speaker_group in context_group:
                if len(speaker_group) > 1:
                    for i_start, i_end in speaker_group:
                        c_id, p_id, s_id = self.abxDataset.get_ids(
                            self.indices_items[i_start]
                        )
                        self.len += (len(speaker_group) - 1) * (
                            min(
                                self.max_x,
                                len(self.get_speakers_from_cp[c_id][p_id]) - 1,
                            )
                        )

    def get_other_speakers_in_group(self, i_start_group):
        c_id, p_id, s_id = self.abxDataset.get_ids(
            self.indices_items[i_start_group]
        )
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
        for i_c, context_group in enumerate(self.indices_item_groups):
            for i_s, speaker_group in enumerate(context_group):
                n_phones = len(speaker_group)
                if n_phones == 1:
                    continue

                for i_a in range(n_phones):
                    i_start_a, i_end_a = self.indices_item_groups[i_c][i_s][i_a]
                    ref = self.get_other_speakers_in_group(i_start_a)
                    if len(ref) > self.max_x:
                        speakers_a = random.sample(ref, k=self.max_x)
                    else:
                        speakers_a = ref

                    for i_start_x, i_end_x in speakers_a:

                        for i_b in range(n_phones):
                            if i_b == i_a:
                                continue

                            i_start_b, i_end_b = self.indices_item_groups[i_c][
                                i_s
                            ][i_b]
                            yield self.get_abx_triplet(
                                (i_start_a, i_end_a),
                                (i_start_b, i_end_b),
                                (i_start_x, i_end_x),
                            )

    def get_board_size(self):

        return (
            self.abxDataset.get_n_speakers(),
            self.abxDataset.get_n_phone(),
            self.abxDataset.get_n_phone(),
            self.abxDataset.get_n_context(),
            self.abxDataset.get_n_speakers(),
        )
