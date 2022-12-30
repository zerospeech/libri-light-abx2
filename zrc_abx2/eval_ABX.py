# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Literal, NamedTuple, Optional, List, Tuple

import torch
import numpy as np
import pandas
from typing_extensions import LiteralString

import zrc_abx2.ABX_src.abx_group_computation as abx_g
from zrc_abx2.ABX_src.ABXDataset.abx_feature_loader import ABXFeatureLoader
from zrc_abx2.ABX_src.ABXIterators.abx_iterator_factory import IteratorFactory
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
# CPC
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
    def eval_abx(self, args: EvalArgs) -> List[Dict[str, Any]]:
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
                    feature_maker,
                    x,
                    strict=args.strict,
                    maxSizeSeq=args.max_size_seq,
                    seqNorm=args.seq_norm,
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

        scores = self._ABX(
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

        return self.formatted_abx_results(scores, args)

    def formatted_abx_results(
        self, scores: Dict[str, float], eval_args: EvalArgs
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for key, val in scores.items():
            result: Dict[str, Any] = {}
            path_data = eval_args.path_data
            item_f = eval_args.path_item_file
            result["path_data"] = path_data
            result["item-file"] = item_f
            try:
                dataset = Path(item_f).stem.split("-")
                result["dataset"] = dataset[0]
                result["sub-dataset"] = dataset[1]
            except:
                warnings.warn(
                    "Unable to retrieve dataset names for the results. Proceeding.",
                    RuntimeWarning,
                )
            result["pooling"] = eval_args.pooling
            result["seed"] = eval_args.seed
            result["run-date"] = datetime.now().strftime("%Y-%m-%d")
            result["score"] = val
            try:
                result["abx-s-condition"] = key.split("-")[0]
                result["abx-c-condition"] = key.split("-")[1]
            except:
                raise ValueError(
                    "Unable to retrieve abx condition definitions for the results."
                )
            results.append(result)
        return results

    def _ABX(
        self,
        pooling: Pooling,
        seed_n: int,
        feature_function: Callable,
        path_item_file: str,
        seq_list: List[Tuple[str, LiteralString]],
        distance_mode: str,
        step_feature: float,
        speakermodes: List[str],
        contextmodes: List[str],
        cuda=False,
        max_x_across=5,
        max_size_group=30,
    ) -> Dict[str, float]:
        # Distance function
        distance_function = abx_g.get_distance_function_from_name(distance_mode)

        # Output
        scores: Dict[str, float] = {}

        print(
            "Date and time of run start:",
            datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

        # ABX calculations differ per context mode
        for contextmode in contextmodes:
            if not contextmode in ("within", "any"):
                raise ValueError(f"Contextmode not supported: {contextmode}")

            ABXDataset = ABXFeatureLoader(
                pooling,
                path_item_file,
                seq_list,
                feature_function,
                step_feature,
                True,
            ).loadFromFileData()

            dimnwithin = None
            dimnacross: list[int] = []
            if contextmode == "within":
                dimnwithin = 3  # TODO: can we make these programmatic?
                dimnacross = [3, 4]
            elif contextmode == "any":
                # dimnwithin not used in this condition.
                dimnacross = [3]

            if cuda:
                ABXDataset.cuda()

            # ABX within speaker
            if "within" in speakermodes:
                print(f"Computing ABX {contextmode} context within speakers...")
                ABXIterator = IteratorFactory.get_iterator(
                    ABXDataset, contextmode, "within", max_size_group, seed_n
                )
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
                    divisor_context = torch.sparse.sum(
                        index_, dimnwithin
                    ).to_dense()
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
                ABXIterator = IteratorFactory.get_iterator(
                    ABXDataset, contextmode, "across", max_size_group, seed_n
                )
                ABXIterator.max_x = max_x_across # Only used in across-speaker
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
                if not dimnacross:
                    raise ValueError("dimnacross not set")
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

    def _find_all_files(
        self, path_dir, extension
    ) -> List[Tuple[str, LiteralString]]:
        """Returns: a list of tuples, each tuple having this format:
        [0]: filename (no extension);
        [1]: absolute path of the file.
        """
        out: List[Tuple[str, LiteralString]] = []
        for root, dirs, filenames in os.walk(path_dir):
            for f in filenames:
                if f.endswith(extension):
                    out.append(((str(Path(f).stem)), os.path.join(root, f)))
        return out

    def _reduce_sparse_data(self, quotient, divisor):
        return quotient / (1e-08 * (divisor == 0) + divisor)

    def _pooling_type(
        self, pooling: str
    ) -> Literal[Pooling.NONE, Pooling.MEAN, Pooling.HAMMING]:
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
        "path_data",
        type=str,
        help="Path to directory containing the submission data",
    )
    parser.add_argument(
        "path_item_file",
        type=str,
        help="Path to the .item file containing the timestamps and transcriptions",
    )
    parser.add_argument(
        "--path_checkpoint",
        type=str,
        default=PATH_CHECKPOINT,
        help="Path to a CPC checkpoint. If set, apply the "
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
        help="When computing the ABX across score, maximum "
        "number of speaker X to sample per couple A,B. "
        " A small value will make the code faster but "
        "less precise.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=OUT,
        help="Path where the results should be saved.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Seed to use in random sampling."
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["none", "mean", "hamming"],
        default=POOLING,
        help="Type of pooling over frame representations of items.",
    )
    parser.add_argument(
        "--seq_norm",
        action="store_true",
        help="Used for CPC features only. "
        "If activated, normalize each batch of feature across the "
        "time channel before computing ABX.",
    )
    parser.add_argument(
        "--max_size_seq",
        default=MAX_SIZE_SEQ,
        type=int,
        help="Used for CPC features only. Maximal number of frames to consider when computing a "
        "batch of features.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Used for CPC features only. "
        "If activated, each batch of feature will contain exactly "
        "max_size_seq frames.",
    )

    # multi-gpu / multi-node
    return parser.parse_args(argv)


# CMDLINE
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)
    eval_args = EvalArgs(
        path_data=args.path_data,
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
        strict=args.strict,
    )
    abx_evaluator = EvalABX()
    scores = abx_evaluator.eval_abx(eval_args)

    if eval_args.out:
        out_dir = Path(eval_args.out)
    elif eval_args.path_checkpoint:
        out_dir = Path(eval_args.path_checkpoint).parent
    else:
        raise ValueError(
            "Unable to find output path from args.out or args.path_checkpoint."
        )
    out_dir.mkdir(exist_ok=True)
    df = pandas.DataFrame(scores)
    with open(out_dir / f"ABX_scores.csv", "a") as file:
        df.to_csv(file, mode="a", index=False, header=file.tell() == 0)

    t = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path_args = out_dir / f"ABX_args_{t}.json"
    with open(path_args, "w") as file:
        json.dump(vars(args), file, indent=2)


if __name__ == "__main__":
    main()
