from ..ABXDataset.abx_feature_dataset import ABXFeatureDataset
from ..ABXIterators.abx_iterators_anycontext import *
from ..ABXIterators.abx_iterators import *
from typing import Union


class IteratorFactory:
    @classmethod
    def get_iterator(
        cls,
        abxDataset: ABXFeatureDataset,
        context_mode: str,
        speaker_mode: str,
        max_size_group: int,
        seed_n: int,
    ) -> Union[
        ABXWithinGroupIterator,
        ABXAcrossGroupIterator
    ]:
        if not (context_mode == "within" or context_mode == "any"):
            raise ValueError(f"Unsupported context mode: {context_mode}")
        if speaker_mode == "within":
            return ABXWithinGroupIterator(
                abxDataset=abxDataset,
                max_size_group=max_size_group,
                reorder_priority=[CONTEXT_IDX, SPEAKER_IDX, PHONE_IDX],
                context_type=ContextType.WITHIN,
                seed_n=seed_n,
            )
        if speaker_mode == "across":
            return ABXAcrossGroupIterator(
                abxDataset=abxDataset,
                max_size_group=max_size_group,
                reorder_priority=[CONTEXT_IDX, SPEAKER_IDX, PHONE_IDX],
                context_type=ContextType.WITHIN,
                seed_n=seed_n,
            )
        if speaker_mode == "any":
            raise ValueError(f"Speaker mode not yet supported: {speaker_mode}")
        raise ValueError(f"Invalid speaker mode: {speaker_mode}")
