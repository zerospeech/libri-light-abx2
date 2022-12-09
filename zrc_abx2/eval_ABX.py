# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import torch
import numpy as np
from dataclasses import dataclass
from typing_extensions import LiteralString

import zrc_abx2.ABX_src.abx_group_computation as abx_g
import zrc_abx2.ABX_src.abx_iterators as abx_it  # <- within context
import zrc_abx2.ABX_src.phone_abx_iterators as phone_abx_it  # <- without context
from zrc_abx2.ABX_src.models import Pooling
from zrc_abx2.cpc.feature_loader import FeatureModule, buildFeature, loadModel

# Default args
PATH_CHECKPOINT = None
FILE_EXTENSION = ".npy"
FEATURE_SIZE = 0.01
CUDA = False
SPEAKER_MODE = "all"
CONTEXT_MODE = "all"
DISTANCE_MODE = "cosine"
MAX_SIZE_GROUP = 10
MAX_X_ACROSS = 5
OUT = None
SEED = 3459
POOLING = "none"
#CPC
SEQ_NORM = False
MAX_SIZE_SEQ = 64000
STRICT = False

class EvalArgs(NamedTuple):
    # See parse_args for help
    # Mandatory
    path_data: str
    path_item_file: str

    # Args with defaults
    path_checkpoint: Optional[str] = PATH_CHECKPOINT
    file_extension: str = FILE_EXTENSION
    feature_size: float = FEATURE_SIZE
    cuda: bool = CUDA
    speaker_mode: str = SPEAKER_MODE
    context_mode: str = CONTEXT_MODE
    distance_mode: str = DISTANCE_MODE
    max_size_group: int = MAX_SIZE_GROUP
    max_x_across: int = MAX_X_ACROSS
    out: Optional[str] = OUT
    seed: int = SEED
    pooling: str = POOLING

    # CPC only
    seq_norm: bool = SEQ_NORM
    max_size_seq: int = MAX_SIZE_SEQ
    strict: bool = STRICT

# Feature-loading functions, one per file format
# If model loaded from checkpoint, procedure specified in eval_abx()
def _load_pt(x):
    data = torch.load(x, "cpu")
    assert len(data.size()) == 2
    return data


def _load_npy(x):
    data = torch.tensor(np.load(x))
    assert len(data.size()) == 2
    return data


def _load_txt(x):
    data = torch.tensor(np.loadtxt(x))
    assert len(data.size()) == 2
    return data

def _loadCPCFeatureMaker(
        CPC_pathCheckpoint,
        encoder_layer=False,
        keepHidden=True,
        gru_level=-1,
        cuda=False,
    ):
        if gru_level and gru_level > 0:
            updateConfig = argparse.Namespace(nLevelsGRU=gru_level)
        else:
            updateConfig = None
        model, _, _ = loadModel([CPC_pathCheckpoint], updateConfig=updateConfig)
        model.gAR.keepHidden = keepHidden
        featureMaker = FeatureModule(model, get_encoded=encoder_layer)
        featureMaker.eval()
        if cuda:
            featureMaker.cuda()

        return featureMaker

class EvalABX:
    # INTERFACE
    def eval_abx(self, args: EvalArgs):
        print("eval_ABX args:")
        print(args)
        if args.path_checkpoint is None:
            if args.file_extension == ".pt":
                feature_function = _load_pt
            elif args.file_extension == ".npy" or args.file_extension == ".npz":
                feature_function = _load_npy
            elif args.file_extension == ".txt":
                feature_function = _load_txt
        else:
            feature_maker = _loadCPCFeatureMaker(
                args.path_checkpoint,
                encoder_layer=False,
                keepHidden=True,
                gru_level=-1,
                cuda=False,
            )

            def feature_function(x):
                return buildFeature(
                    feature_maker, x, strict=args.strict, maxSizeSeq=args.max_size_seq, seqNorm=args.seq_norm, 
                )[0]

        # Speaker modes
        if args.speaker_mode == "all":
            speakermodes = ["within", "across"]
        else:
            speakermodes = [args.speaker_mode]

        # Context modes
        if args.context_mode == "all":
            contextmodes = ["within", "any"]
        else:
            contextmodes = [args.context_mode]

        step_feature = 1 / args.feature_size

        # Get the list of sequences
        seq_list = self._find_all_files(args.path_data, args.file_extension)

        return self._ABX(
            self._pooling_type(args.pooling),
            args.seed,
            feature_function,
            args.path_item_file,
            seq_list,
            args.distance_mode,
            step_feature,
            speakermodes,
            contextmodes,
            cuda=args.cuda,
            max_x_across=args.max_x_across,
            max_size_group=args.max_size_group,
        )

    def _ABX(
        self,
        pooling: Pooling,
        seed_n: int,
        feature_function: Callable,
        path_item_file: str,
        seq_list: list[tuple[str, LiteralString]],
        distance_mode: str,
        step_feature: float,
        speakermodes: list[str],
        contextmodes: list[str],
        cuda=False,
        max_x_across=5,
        max_size_group=30,
    ):
        # Distance function
        distance_function = abx_g.get_distance_function_from_name(distance_mode)

        # Output
        scores = {}

        print(
            "Date and time of run start:", datetime.now().strftime("%d/%m/%Y %H:%M")
        )
        # ABX calculations differ per context mode
        for contextmode in contextmodes:

            # ABXDataset at present depends on the contextmode
            # (as does the dimension value used for some arrays)
            # This may change with better streamlining
            if contextmode == "within":
                ABXDataset = abx_it.ABXFeatureLoader(
                    pooling,
                    seed_n,
                    path_item_file,
                    seq_list,
                    feature_function,
                    step_feature,
                    True,
                )
                dimnwithin = 3
                dimnacross = [3, 4]
            elif contextmode == "any":
                ABXDataset = phone_abx_it.phoneABXFeatureLoader(
                    pooling,
                    seed_n,
                    path_item_file,
                    seq_list,
                    feature_function,
                    step_feature,
                    True,
                )
                dimnwithin = None  # not actually used
                dimnacross = [3]

            if cuda:
                ABXDataset.cuda()

            # ABX within
            if "within" in speakermodes:
                print(f"Computing ABX {contextmode} context within speakers...")
                ABXIterator = ABXDataset.get_iterator("within", max_size_group)
                group_confusion = abx_g.get_abx_scores_dtw_on_group(
                    ABXIterator, distance_function, ABXIterator.symmetric, pooling
                )

                n_data = group_confusion._values().size(0)
                index_ = torch.sparse.LongTensor(
                    group_confusion._indices(),
                    torch.ones((n_data), dtype=torch.float),
                    group_confusion.size(),
                )
                if contextmode == "any":
                    divisor_context = index_.to_dense()
                    group_confusion = group_confusion.to_dense()
                else:
                    divisor_context = torch.sparse.sum(index_, dimnwithin).to_dense()
                    group_confusion = torch.sparse.sum(
                        group_confusion, dimnwithin
                    ).to_dense()
                    group_confusion = self._reduce_sparse_data(
                        group_confusion, divisor_context
                    )

                S, p1, p2 = group_confusion.size()

                index_speaker = divisor_context > 0
                divisor_speaker = index_speaker.sum(dim=0)
                phone_confusion = self._reduce_sparse_data(
                    group_confusion.sum(dim=0), divisor_speaker
                )

                scores[f"within-{contextmode}"] = (
                    phone_confusion.sum() / (divisor_speaker > 0).sum()
                ).item()
                print(
                    f"...done. ABX {contextmode}_context within_speaker : {scores[f'within-{contextmode}']}"
                )

            # ABX across
            if "across" in speakermodes:
                print(f"Computing ABX {contextmode} context across speakers...")
                ABXIterator = ABXDataset.get_iterator("across", max_size_group)
                ABXIterator.max_x = max_x_across
                group_confusion = abx_g.get_abx_scores_dtw_on_group(
                    ABXIterator, distance_function, ABXIterator.symmetric, pooling
                )
                n_data = group_confusion._values().size(0)
                index_ = torch.sparse.LongTensor(
                    group_confusion._indices(),
                    torch.ones((n_data), dtype=torch.float),
                    group_confusion.size(),
                )
                divisor_context = torch.sparse.sum(index_, dimnacross).to_dense()
                group_confusion = torch.sparse.sum(
                    group_confusion, dimnacross
                ).to_dense()
                group_confusion = self._reduce_sparse_data(
                    group_confusion, divisor_context
                )
                S, p1, p2 = group_confusion.size()

                index_speaker = divisor_context > 0
                divisor_speaker = index_speaker.sum(dim=0)
                phone_confusion = self._reduce_sparse_data(
                    group_confusion.sum(dim=0), divisor_speaker
                )
                scores[f"across-{contextmode}"] = (
                    phone_confusion.sum() / (divisor_speaker > 0).sum()
                ).item()
                print(
                    f"...done. ABX {contextmode}_context across_speaker : {scores[f'across-{contextmode}']}"
                )

        return scores


    def _find_all_files(self, path_dir, extension) -> list[tuple[str, LiteralString]]:
        """Returns: a list of tuples, each tuple having this format:
        [0]: filename (no extension);
        [1]: absolute path of the file.
        """
        out: list[tuple[str, LiteralString]] = []
        for root, dirs, filenames in os.walk(path_dir):
            for f in filenames:
                if f.endswith(extension):
                    out.append(((str(Path(f).stem)), os.path.join(root, f)))
        return out


    def _reduce_sparse_data(self, quotient, divisor):
        return quotient / (1e-08 * (divisor == 0) + divisor)

    def _pooling_type(self, pooling: str):
        if pooling == "none":
            return Pooling.NONE
        elif pooling == "mean":
            return Pooling.MEAN
        elif pooling == "hamming":
            return Pooling.HAMMING
        else:
            raise ValueError("Unsupported pooling type.")


def parse_args(argv):

    parser = argparse.ArgumentParser(description="ABX metric")

    parser.add_argument(
        "path_data", type=str, help="Path to directory containing the data"
    )
    parser.add_argument("path_item_file", type=str, help="Path to the .item file")
    parser.add_argument(
        "--path_checkpoint",
        type=str,
        default=PATH_CHECKPOINT,
        help="Path to a CPC checkpoint. If set, the apply the "
        "model to the input data to compute the features",
    )
    parser.add_argument(
        "--file_extension",
        type=str,
        default=FILE_EXTENSION,
        choices=[".pt", ".npy", ".wav", ".flac", ".mp3", ".npz", ".txt"],
    )
    parser.add_argument(
        "--feature_size",
        type=float,
        default=FEATURE_SIZE,
        help="Size (in s) of one feature",
    )
    parser.add_argument(
        "--cuda", action="store_true", help="Use the GPU to compute distances"
    )
    parser.add_argument(
        "--speaker_mode",
        type=str,
        default=SPEAKER_MODE,
        choices=["all", "within", "across"],
        help="Choose the speaker mode of the ABX score to compute",
    )
    parser.add_argument(
        "--context_mode",
        type=str,
        default=CONTEXT_MODE,
        choices=["all", "within", "any"],
        help="Choose the context mode of the ABX score to compute",
    )
    parser.add_argument(
        "--distance_mode",
        type=str,
        default=DISTANCE_MODE,
        choices=["euclidian", "euclidean", "cosine", "kl", "kl_symmetric"],
        help="Choose the kind of distance to use to compute " "the ABX score.",
    )
    parser.add_argument(
        "--max_size_group",
        type=int,
        default=MAX_SIZE_GROUP,
        help="Max size of a group while computing the"
        "ABX score. A small value will make the code "
        "faster but less precise.",
    )
    parser.add_argument(
        "--max_x_across",
        type=int,
        default=MAX_X_ACROSS,
        help="When computing the ABX across score, maximum"
        "number of speaker X to sample per couple A,B. "
        " A small value will make the code faster but "
        "less precise.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=OUT,
        help="Path where the results should be saved",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed to use in random sampling.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["none", "mean", "hamming"],
        default=POOLING,
        help="Type of pooling over frame representations of items.",
    )
    parser.add_argument(
        '--seq_norm',
        action='store_true',
        help='Used for CPC features only. '
        'If activated, normalize each batch of feature across the '
        'time channel before computing ABX.',
    )
    parser.add_argument(
        '--max_size_seq',
        default=MAX_SIZE_SEQ,
        type=int,
        help='Used for CPC features only. Maximal number of frames to consider when computing a '
        'batch of features.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Used for CPC features only. '
        'If activated, each batch of feature will contain exactly '
        'max_size_seq frames.',
    )

    # multi-gpu / multi-node
    return parser.parse_args(argv)

# CMDLINE
def main(argv):
    args = parse_args(argv)
    eval_args = EvalArgs(path_data=args.path_data, 
                    path_item_file=args.path_item_file, 
                    path_checkpoint=args.path_checkpoint,
                    file_extension=args.file_extension,
                    feature_size=args.feature_size,
                    cuda=args.cuda, 
                    speaker_mode=args.speaker_mode,
                    context_mode=args.context_mode,
                    distance_mode=args.distance_mode,
                    max_x_across=args.max_x_across,
                    out=args.out,
                    seed=args.seed,
                    pooling=args.pooling,
                    seq_norm=args.seq_norm,
                    max_size_seq=args.max_size_seq,
                    strict=args.strict)
    scores = EvalABX().eval_abx(eval_args)
    if eval_args.out:
        out_dir = Path(eval_args.out)
    elif eval_args.path_checkpoint:
        out_dir = Path(eval_args.path_checkpoint).parent
    else:
        raise ValueError("Unable to find output path from args.out or args.path_checkpoint.")
    out_dir.mkdir(exist_ok=True)

    path_score = out_dir / f"ABX_scores_pooling-{args.pooling}.json"
    with open(path_score, "w") as file:
        json.dump(scores, file, indent=2)

    path_args = out_dir / f"ABX_args_pooling-{args.pooling}.json"
    with open(path_args, "w") as file:
        json.dump(vars(args), file, indent=2)

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)