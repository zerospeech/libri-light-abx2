# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import sys
import torch
import json
import os
import numpy as np
import ABX_src.abx_group_computation as abx_g
import ABX_src.abx_iterators as abx_it  # <- within context
import ABX_src.phone_abx_iterators as phone_abx_it  # <- without context
from cpc.feature_loader import buildFeature, loadCPCFeatureMaker
from pathlib import Path


def find_all_files(path_dir, extension):
    out = []
    for root, dirs, filenames in os.walk(path_dir):
        for f in filenames:
            if f.endswith(extension):
                out.append(((str(Path(f).stem)), os.path.join(root, f)))
    return out


def reduce_sparse_data(quotient, divisor):
    return quotient / (1e-08 * (divisor == 0) + divisor)


# Feature-loading functions, one per file format
def load_pt(x):
    data = torch.load(x, "cpu")
    assert len(data.size()) == 2
    return data


def load_npy(x):
    data = torch.tensor(np.load(x))
    assert len(data.size()) == 2
    return data


def load_txt(x):
    data = torch.tensor(np.loadtxt(x))
    assert len(data.size()) == 2
    return data


# If model loaded from checkpoint, procedure specified in main() below


def ABX(
    feature_function,
    path_item_file,
    seq_list,
    distance_mode,
    step_feature,
    speakermodes,
    contextmodes,
    cuda=False,
    max_x_across=5,
    max_size_group=30,
):

    # Distance function
    distance_function = abx_g.get_distance_function_from_name(distance_mode)

    # Output
    scores = {}

    # ABX calculations differ per context mode
    for contextmode in contextmodes:

        # ABXDataset at present depends on the contextmode
        # (as does the dimension value used for some arrays)
        # This may change with better streamlining
        if contextmode == "within":
            ABXDataset = abx_it.ABXFeatureLoader(
                path_item_file, seq_list, feature_function, step_feature, True
            )
            dimnwithin = 3
            dimnacross = [3, 4]
        elif contextmode == "any":
            ABXDataset = phone_abx_it.phoneABXFeatureLoader(
                path_item_file, seq_list, feature_function, frame_step, True
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
                ABXIterator, distance_function, ABXIterator.symmetric
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
                group_confusion = reduce_sparse_data(
                    group_confusion, divisor_context
                )

            S, p1, p2 = group_confusion.size()

            index_speaker = divisor_context > 0
            divisor_speaker = index_speaker.sum(dim=0)
            phone_confusion = reduce_sparse_data(
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
                ABXIterator, distance_function, ABXIterator.symmetric
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
            group_confusion = reduce_sparse_data(
                group_confusion, divisor_context
            )
            S, p1, p2 = group_confusion.size()

            index_speaker = divisor_context > 0
            divisor_speaker = index_speaker.sum(dim=0)
            phone_confusion = reduce_sparse_data(
                group_confusion.sum(dim=0), divisor_speaker
            )
            scores[f"across-{contextmode}"] = (
                phone_confusion.sum() / (divisor_speaker > 0).sum()
            ).item()
            print(
                f"...done. ABX {contextmode}_context across_speaker : {scores[f'across-{contextmode}']}"
            )

    return scores


def parse_args(argv):

    parser = argparse.ArgumentParser(description="ABX metric")

    parser.add_argument(
        "path_data", type=str, help="Path to directory containing the data"
    )
    parser.add_argument(
        "path_item_file", type=str, help="Path to the .item file"
    )
    parser.add_argument(
        "--path_checkpoint",
        type=str,
        default=None,
        help="Path to a CPC checkpoint. If set, the apply the "
        "model to the input data to compute the features",
    )
    parser.add_argument(
        "--file_extension",
        type=str,
        default=".pt",
        choices=[".pt", ".npy", ".wav", ".flac", ".mp3", ".npz"],
    )
    parser.add_argument(
        "--feature_size",
        type=float,
        default=0.01,
        help="Size (in s) of one feature",
    )
    parser.add_argument(
        "--cuda", action="store_true", help="Use the GPU to compute distances"
    )
    parser.add_argument(
        "--speaker_mode",
        type=str,
        default="all",
        choices=["all", "within", "across"],
        help="Choose the speaker mode of the ABX score to compute",
    )
    parser.add_argument(
        "--context_mode",
        type=str,
        default="all",
        choices=["all", "within", "any"],
        help="Choose the context mode of the ABX score to compute",
    )
    parser.add_argument(
        "--distance_mode",
        type=str,
        default="cosine",
        choices=["euclidian", "cosine", "kl", "kl_symmetric"],
        help="Choose the kind of distance to use to compute " "the ABX score.",
    )
    parser.add_argument(
        "--max_size_group",
        type=int,
        default=10,
        help="Max size of a group while computing the"
        "ABX score. A small value will make the code "
        "faster but less precise.",
    )
    parser.add_argument(
        "--max_x_across",
        type=int,
        default=5,
        help="When computing the ABX across score, maximum"
        "number of speaker X to sample per couple A,B. "
        " A small value will make the code faster but "
        "less precise.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path where the results should be saved",
    )

    # multi-gpu / multi-node
    return parser.parse_args(argv)


def main(argv):

    args = parse_args(argv)

    print(args)

    if args.path_checkpoint is None:
        if args.file_extension == ".pt":
            feature_function = load_pt
        elif args.file_extension == ".npy" or args.file_extension == ".npz":
            feature_function = load_npy
        elif args.file_extension == ".txt":
            feature_function = load_txt
    else:
        feature_maker, _ = loadCPCFeatureMaker(
            args.path_checkpoint,
            encoder_layer=False,
            keepHidden=True,
            gru_level=-1,
            cuda=False,
        )

        def feature_function(x):
            return buildFeature(
                feature_maker, x, seqNorm=False, strict=True, maxSizeSeq=64000
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
    seq_list = find_all_files(args.path_data, args.file_extension)

    scores = ABX(
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

    out_dir = (
        Path(args.path_checkpoint).parent
        if args.out is None
        else Path(args.out)
    )
    out_dir.mkdir(exist_ok=True)

    path_score = out_dir / "ABX_scores.json"
    with open(path_score, "w") as file:
        json.dump(scores, file, indent=2)

    path_args = out_dir / "ABX_args.json"
    with open(path_args, "w") as file:
        json.dump(vars(args), file, indent=2)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
