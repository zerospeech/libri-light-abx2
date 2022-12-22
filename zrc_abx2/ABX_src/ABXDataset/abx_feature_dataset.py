from dataclasses import dataclass
from typing import Any, List, Tuple

import torch

from .abx_item_file_loader import *
from ..models import *


@dataclass
class ABXFeatureDataset:
    # A list of all the frames representations over all the items
    # Each row is a representation of a frame
    data: torch.Tensor
    # The manifest lists where in data each item is along with the
    # encoded phone, context, and speaker ids from the transcription
    features_manifest: List[ManifestFeatureItem]
    feature_dim: Any  # TODO: Can we specify the type?
    item_file: ItemFile

    def get_ids(self, index) -> Tuple[int, int, int]:
        phone_id = self.features_manifest[index][PHONE_IDX]
        speaker_id = self.features_manifest[index][SPEAKER_IDX]
        context_id = self.features_manifest[index][CONTEXT_IDX]
        return context_id, phone_id, speaker_id

    def __len__(self) -> int:
        return len(self.features_manifest)

    def __getitem__(self, index) -> ABXFeaturesDataItem:
        (
            i_data,
            out_size,
            context_id,
            phone_id,
            speaker_id,
        ) = self.features_manifest[index]
        return ABXFeaturesDataItem(
            self.data[i_data : (i_data + out_size)],
            out_size,
            context_id,
            phone_id,
            speaker_id,
        )

    def get_data_device(self) -> torch.device:
        return self.data.device

    def cuda(self):
        self.data = self.data.cuda()

    def cpu(self):
        self.data = self.data.cpu()

    def get_n_speakers(self) -> int:
        return len(self.item_file.speaker_match)

    def get_n_context(self) -> int:
        return len(self.item_file.context_match)

    def get_n_phone(self) -> int:
        return len(self.item_file.phone_match)
