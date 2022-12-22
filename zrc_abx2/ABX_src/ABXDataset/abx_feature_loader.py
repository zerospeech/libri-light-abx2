import math
import random
from typing import Any, Callable

import torch
import numpy as np
from typing_extensions import LiteralString

from ABX_src.ABXDataset.abx_feature_dataset import ABXFeatureDataset
from ABX_src.ABXDataset.abx_item_file_loader import *
from ABX_src.models import *


def normalize_with_singularity(x) -> torch.Tensor:
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
    border_vect = torch.zeros((S, 1), dtype=x.dtype, device=x.device) + 1e-12
    border_vect[zero_vals] = -2 * 1e12
    return torch.cat([x, border_vect], dim=1)


class ABXFeatureLoader:
    def __init__(
        self,
        pooling: Pooling,
        path_item_file: str,
        seqList: list[tuple[str, LiteralString]],
        feature_maker: Callable,
        stepFeature: float,
        normalize: bool,
    ):
        """
        Args:
            path_item_file (str): path to the .item (.pitem) files containing the ABX
                                  triplets
            seqList (list): list of files (fileID, path) where fileID refers to
                            the file's ID as used in (path_)item_file, and path
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

        self.pooling = pooling
        self.item_file = ABXItemFileLoader().load_item_file(path_item_file)
        self.seqList = seqList
        self.feature_maker = feature_maker
        self.stepFeature = stepFeature
        self.normalize = normalize
        self.seqNorm = True  

    # INTERFACE
    def loadFromFileData(self) -> ABXFeatureDataset:
        return self._load_data(
            self.pooling,
            self.item_file.files_data,
            self.seqList,
            self.feature_maker,
            self.normalize,
            self.stepFeature,
            self.item_file,
        )

    # PRIVATE METHODS
    def _pool(self, feature: torch.Tensor, pooling: Pooling) -> torch.Tensor:
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

    def _start_end_indices(
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

    def _append_feature(
        self,
        index_start: int,
        index_end: int,
        totSize: int,
        all_features: torch.Tensor,
        context_id: int,
        phone_id: int,
        speaker_id: int,
        data: list[torch.Tensor],
        features_manifest: list[ManifestFeatureItem],
        pooling: Pooling,
    ) -> int:
        """Build and append the feature to the features data list.
        Add information on it to the manifest, i.e. to self.features.
        Return the total size i.e. the total number of frames added to the data thus far."""
        feature = all_features[index_start:index_end]
        feature = self._pool(feature, pooling)
        start_i = totSize
        loc_size = feature.shape[0]
        features_manifest.append(
            ManifestFeatureItem(
                start_i, loc_size, context_id, phone_id, speaker_id
            )
        )
        data.append(feature)
        return totSize + loc_size

    def _load_data(
        self,
        pooling: Pooling,
        files_data: dict[str, list[ItemData]],
        seqList: list[tuple[str, LiteralString]],
        feature_maker: Callable,
        normalize: bool,
        stepFeature: float,
        item_file: ItemFile,
    ) -> ABXFeatureDataset:
        # data[i] is the data for a given item.
        # data contains all the item representations over all files
        data: list[torch.Tensor] = []
        # features_manifest[i]: index_start, size, context_id, phone_id,
        # speaker_id. This is a manifest of what is in
        # data_compressed (see below)
        features_manifest: list[ManifestFeatureItem] = []

        totSize = 0

        print("Building the input features...")

        for vals in seqList:
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

            for item_data in phone_data:
                index_start, index_end = self._start_end_indices(
                    item_data.onset,
                    item_data.offset,
                    all_features,
                    stepFeature,
                )
                if (
                    index_start >= all_features.size(0)
                    or index_end <= index_start
                ):
                    continue
                totSize = self._append_feature(
                    index_start,
                    index_end,
                    totSize,
                    all_features,
                    item_data.context_id,
                    item_data.phone_id,
                    item_data.speaker_id,
                    data,
                    features_manifest,
                    pooling,
                )

        print("...done")
        # A list of all the frames representations over all the items
        # such that first we have the frame representations from the
        # first item in data; then, all the frame representations from
        # the second item in data get concatenated to this, etc.
        data_compressed = torch.cat(data, dim=0)
        feature_dim = data_compressed.size(1)
        return ABXFeatureDataset(
            data_compressed, features_manifest, feature_dim, item_file
        )
