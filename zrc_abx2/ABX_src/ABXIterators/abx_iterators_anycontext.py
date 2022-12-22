# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


# This version disregards the context parameters, i.e. to run
# "without-context".

# It supports across- and within-speaker ABX just like the original.

import random

from .abx_iterators import ABXIterator
from ..ABXDataset.abx_feature_dataset import ABXFeatureDataset
from ..models import *


## ITERATORS THAT IGNORE CONTEXT
class ABXWithinGroupIteratorAnyContext(ABXIterator):
    r"""
    Iterator giving the triplets for the ABX within speaker score.
    """

    def __init__(
        self,
        abxDataset: ABXFeatureDataset,
        max_size_group,
        reorder_priority: List[int],
        context_type: ContextType,
        seed_n: int
    ):

        super().__init__(
            abxDataset=abxDataset,
            max_size_group=max_size_group,
            symmetric=True,
            reorder_priority=reorder_priority,
            context_type=context_type,
            seed_n=seed_n
        )

        for speaker_group in self.indices_item_groups:
            if len(speaker_group) > 1:
                for i_start, i_end in speaker_group:
                    if i_end - i_start > 1:
                        self.len += len(speaker_group) - 1

    def __iter__(self):
        for i_s, speaker_group in enumerate(self.indices_item_groups):
            n_phones = len(speaker_group)
            if n_phones == 1:
                continue

            for i_a in range(n_phones):
                i_start_a, i_end_a = self.indices_item_groups[i_s][i_a]
                if i_end_a - i_start_a == 1:
                    continue

                for i_b in range(n_phones):
                    if i_b == i_a:
                        continue

                    i_start_b, i_end_b = self.indices_item_groups[i_s][i_b]
                    data_b, size_b, id_b = self.get_group(i_start_b, i_end_b)
                    data_a, size_a, id_a = self.get_group(i_start_a, i_end_a)

                    out_coords = id_a[1], id_a[0], id_b[0]
                    yield out_coords, (data_a, size_a), (data_b, size_b), (
                        data_a,
                        size_a,
                    )

    def get_board_size(self):

        return (
            self.abxDataset.get_n_speakers(),
            self.abxDataset.get_n_phone(),
            self.abxDataset.get_n_phone(),
        )


class ABXAcrossGroupIteratorAnyContext(ABXIterator):
    r"""
    Iterator giving the triplets for the ABX across score.
    """

    def __init__(
        self,
        abxDataset: ABXFeatureDataset,
        max_size_group,
        reorder_priority: List[int],
        context_type: ContextType,
        seed_n: int
    ):

        super().__init__(
            abxDataset=abxDataset,
            max_size_group=max_size_group,
            symmetric=False,
            reorder_priority=reorder_priority,
            context_type=context_type,
            seed_n=seed_n
        )
        self.get_speakers_from_p = {}
        self.max_x = 5

        for speaker_group in self.indices_item_groups:
            for i_start, i_end in speaker_group:
                _, p_id, s_id = self.abxDataset.get_ids(
                    self.indices_items[i_start]
                )  # Different from original
                if p_id not in self.get_speakers_from_p:
                    self.get_speakers_from_p[p_id] = {}
                self.get_speakers_from_p[p_id][s_id] = (i_start, i_end)

        for speaker_group in self.indices_item_groups:
            if len(speaker_group) > 1:
                for i_start, i_end in speaker_group:
                    _, p_id, s_id = self.abxDataset.get_ids(
                        self.indices_items[i_start]
                    )  # Different from original
                    self.len += (len(speaker_group) - 1) * (
                        min(self.max_x, len(self.get_speakers_from_p[p_id]) - 1)
                    )

    def get_other_speakers_in_group(self, i_start_group):
        _, p_id, s_id = self.abxDataset.get_ids(
            self.indices_items[i_start_group]
        )  # Different from original
        return [v for k, v in self.get_speakers_from_p[p_id].items() if k != s_id]

    def get_abx_triplet(self, i_a, i_b, i_x):
        i_start_a, i_end_a = i_a
        data_a, size_a, id_a = self.get_group(i_start_a, i_end_a)

        i_start_b, i_end_b = i_b
        data_b, size_b, id_b = self.get_group(i_start_b, i_end_b)

        i_start_x, i_end_x = i_x
        data_x, size_x, id_x = self.get_group(i_start_x, i_end_x)

        out_coords = id_a[1], id_a[0], id_b[0], id_x[1]
        return out_coords, (data_a, size_a), (data_b, size_b), (data_x, size_x)

    def __iter__(self):
        for i_s, speaker_group in enumerate(self.indices_item_groups):
            n_phones = len(speaker_group)
            if n_phones == 1:
                continue

            for i_a in range(n_phones):
                i_start_a, i_end_a = self.indices_item_groups[i_s][i_a]
                ref = self.get_other_speakers_in_group(i_start_a)
                if len(ref) > self.max_x:
                    speakers_a = random.sample(ref, k=self.max_x)
                else:
                    speakers_a = ref

                for i_start_x, i_end_x in speakers_a:

                    for i_b in range(n_phones):
                        if i_b == i_a:
                            continue

                        i_start_b, i_end_b = self.indices_item_groups[i_s][i_b]
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
            self.abxDataset.get_n_speakers(),
        )