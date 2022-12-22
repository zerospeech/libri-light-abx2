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
        ABXAcrossGroupIterator,
        ABXWithinGroupIteratorAnyContext,
        ABXAcrossGroupIteratorAnyContext,
    ]:
        if context_mode == "within":
            retriever = cls.get_iterator_within_context
        elif context_mode == "any":
            retriever = cls.get_iterator_any_context
        else:
            raise ValueError(f"Invalid mode: {context_mode}")
        return retriever(abxDataset, speaker_mode, max_size_group, seed_n)

    @classmethod
    def get_iterator_within_context(
        cls,
        abxDataset: ABXFeatureDataset,
        speaker_mode: str,
        max_size_group: int,
        seed_n: int,
    ) -> Union[ABXWithinGroupIterator, ABXAcrossGroupIterator]:
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
            raise ValueError(f"Mode not yet supported: {speaker_mode}")
        raise ValueError(f"Invalid mode: {speaker_mode}")

    @classmethod
    def get_iterator_any_context(
        cls,
        abxDataset: ABXFeatureDataset,
        speaker_mode: str,
        max_size_group: int,
        seed_n: int,
    ) -> Union[
        ABXWithinGroupIteratorAnyContext, ABXAcrossGroupIteratorAnyContext
    ]:
        if speaker_mode == "within":
            return ABXWithinGroupIteratorAnyContext(
                abxDataset=abxDataset,
                max_size_group=max_size_group,
                reorder_priority=[SPEAKER_IDX, PHONE_IDX],
                context_type=ContextType.ANY,
                seed_n=seed_n,
            )
        if speaker_mode == "across":
            return ABXAcrossGroupIteratorAnyContext(
                abxDataset=abxDataset,
                max_size_group=max_size_group,
                reorder_priority=[SPEAKER_IDX, PHONE_IDX],
                context_type=ContextType.ANY,
                seed_n=seed_n,
            )
        raise ValueError(f"Invalid mode: {speaker_mode}")
