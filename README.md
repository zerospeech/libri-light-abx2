# libri-light-abx2

The ABX phonetic evaluation metric for unsupervised representation learning as used by the ZeroSpeech challenge, now with context-type options (on-triphone, within-context, any-context). This module is a reworking of https://github.com/zerospeech/libri-light-abx, which in turn is a wrapper around https://github.com/facebookresearch/libri-light/tree/main/eval

  
### Installation
  
You can install this module from pip directly using the following command : 

`pip install zerospeech-libriabx2`

Or you can install from source by cloning this repository and running: 

`pip install .`

### Usage
### From command line

```
usage: zrc-abx2 [-h] [--path_checkpoint PATH_CHECKPOINT]
                [--file_extension {.pt,.npy,.wav,.flac,.mp3,.npz,.txt}]
                [--feature_size FEATURE_SIZE] [--cuda]
                [--speaker_mode {all,within,across}]
                [--context_mode {all,within,any}]
                [--distance_mode {euclidian,euclidean,cosine,kl,kl_symmetric}]
                [--max_size_group MAX_SIZE_GROUP]
                [--max_x_across MAX_X_ACROSS] [--out OUT] [--seed SEED]
                [--pooling {none,mean,hamming}] [--seq_norm]
                [--max_size_seq MAX_SIZE_SEQ] [--strict]
                path_data path_item_file

ABX metric

positional arguments:
  path_data             Path to directory containing the submission data
  path_item_file        Path to the .item file containing the timestamps and
                        transcriptions

optional arguments:
  -h, --help            show this help message and exit
  --path_checkpoint PATH_CHECKPOINT
                        Path to a CPC checkpoint. If set, apply the model to
                        the input data to compute the features
  --file_extension {.pt,.npy,.wav,.flac,.mp3,.npz,.txt}
  --feature_size FEATURE_SIZE
                        Size (in s) of one feature
  --cuda                Use the GPU to compute distances
  --speaker_mode {all,within,across}
                        Choose the speaker mode of the ABX score to compute
  --context_mode {all,within,any}
                        Choose the context mode of the ABX score to compute
  --distance_mode {euclidian,euclidean,cosine,kl,kl_symmetric}
                        Choose the kind of distance to use to compute the ABX
                        score.
  --max_size_group MAX_SIZE_GROUP
                        Max size of a group while computing the ABX score. A
                        small value will make the code faster but less
                        precise.
  --max_x_across MAX_X_ACROSS
                        When computing the ABX across score, maximum number of
                        speaker X to sample per couple A,B. A small value will
                        make the code faster but less precise.
  --out OUT             Path where the results should be saved
  --seed SEED           Seed to use in random sampling.
  --pooling {none,mean,hamming}
                        Type of pooling over frame representations of items.
  --seq_norm            Used for CPC features only. If activated, normalize
                        each batch of feature across the time channel before
                        computing ABX.
  --max_size_seq MAX_SIZE_SEQ
                        Used for CPC features only. Maximal number of frames
                        to consider when computing a batch of features.
  --strict              Used for CPC features only. If activated, each batch
                        of feature will contain exactly max_size_seq frames.
```
### Python API
You can also call the abx evaluation from python code. You can use the following example:

```
import zrc_abx2

args = zrc_abx2.EvalArgs(
    path_data= "/location/to/representations/",
    path_item_file= "/location/to/file.item",
    **other_options
)

result = zrc_abx2.EvalABX().eval_abx(args)
```

## Information on evaluation conditions
A new  variable in this ABX version is context.
In the within-context condition, a, b, and x have the same surrounding context (i.e. the same preceding and following phoneme). any-context ignores the surrounding context; typically, it varies. 

For the within-context and any-context comparison, use an item file that extracts phonemes (rather than XYZ triphones). For the on-triphone condition, which is still available, use an item file that extracts triphones (just like in the previous abx evaluation), and then run it within-context (which was the default behavior of the previous abx evaluation). any-context is not used for the on-triphone version due to excessive noise that would be included in the representation.

Like in the previous version, it is also possible to run within-speaker (a, b, x are all from the same speaker) and across-speaker (a and b are from the same speaker, x is from another) evaluations. So there are four phoneme-based evaluation combinations in total: within_s-within_c, within_s-any-c, across_s-within_c, across_s-any_c; and two triphone-based evaluation combinations: within_s-within_c, across_s-within_c. 

