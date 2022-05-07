# ABX_revamped

The ABX phonetic evaluation as used by ZeroSpeech challenge, in the process of being revamped to add "context-type" options (on-triphone, within-context, any-context).  
  
### Requirements?  
  
The requirements for this are the same as for ZS2021 - the environment "zr2021-eval" should be available (???) through oberon, which should include the CPC loading nightmare

### How to use this for CPC checkpoints?

The ZS Challenge ABX evaluation is scattered across three layers: `evaluate.py` specifies the dataset (dev/test), `phonetic.py` specifies the subdataset (clean/other), and once that info is combined, `eval_ABX.py` can take the specific dataset-subdataset combination and runs the ABX evaluation on specified speaker and context modes. 

At present, the ZS version of ABX doesn't accept CPC checkpoints at the upper two layers (`evaluate.py` or `phonetic.py`).  
`eval_ABX.py` on its own *does* accept CPC checkpoints - this is just not carried up through the other two layers.  
The short-term solution for this intermediate "revamping" step is, basically, **to run `eval_ABX.py` four times**: one for each of the dataset-subdataset combinations, `{dev, test} x {clean, other}`.  
  
To run `eval_ABX.py` in isolation from its outer layers, you will need to give it the following arguments:  
  
* `path_data`: path to the data, directory of flac files in this case - there are 4 of these  
    * /scratch1/data/raw_data/LibriSpeech/dev-clean/
    * /scratch1/data/raw_data/LibriSpeech/dev-other/
    * /scratch1/data/raw_data/LibriSpeech/test-clean/
    * /scratch1/data/raw_data/LibriSpeech/test-other/
* `path_item_file`: path to the location of a corresponding textfile - there are accordingly 4 of these as well
    * /scratch2/alyashenko/item_from_alignment/dev-clean/dev-clean.item
    * /scratch2/alyashenko/item_from_alignment/dev-other/dev-other.item
    * /scratch2/alyashenko/item_from_alignment/test-clean/test-clean.item
    * /scratch2/alyashenko/item_from_alignment/test-other/test-other.item  
* **the two above arguments must match when passed to eval_ABX: dev-clean with dev-clean, test-other with test-other, etc.**
* `path_checkpoint`: path to your CPC checkpoint
* `file_extension`: ".flac" in this case (the files in path_data)
* `feature_size` *(optional): size of a single feature, converts to frame step; default will be 100ms*  
* `speaker_mode` & `context_mode` *(optional)*  
    * *these both default to "all", i.e. {"within_s", "across_s"} and {"within_c", "without_c"}*  
    * *you could specify just one of the two options per mode if you wanted, e.g. speaker_mode="across_s" contextmode="without_c"*
* `distance_mode` *(optional): this defaults to "cosine"; other options are 'euclidian', 'kl', 'kl_symmetric'*

In total, the basic run would look as follows:  
  
    cd abx_revamped
    conda activate zr2021-eval
    eval_ABX.py path_data="/scratch1/.../dev-clean/" path_item_file="/scratch2/.../dev-clean.item" path_checkpoint="/.../abc.pt" file_extension=".flac" 

## What this was based on

This repo is a reworking of the ABX phonetic evaluation used by ZS2021 - you can find the bulk of it [here](https://github.com/zerospeech/zerospeech2021/tree/65ba7cbb642a1d56282e7d1b86a728e09a9d6dc5/zerospeech2021). 
The ABX evaluation supports two **speaker modes**: "within" speaker and "across" speaker.

`abx_revamped` mirrors many parts of the `phonetic_eval` directory, which in turn takes after the [libri-light ABX evaluation](https://github.com/facebookresearch/libri-light/tree/main/eval). This version ran on-triphone (the models were given an entire XYZ sequence)
The parts of `abx_revamped` that aren't from ZS2021 or libri-light were borrowed from an on-phone version of ABX (the models were given Y, without X or Z). Hence the name `phone_abx_iterators` and its constituents. 

`abx_revamped` also takes from ZS2021 the file `phonetic.py`, which uses `phonetic_eval`'s contents in ZS-specific ways. [evaluate.py](https://github.com/zerospeech/zerospeech2021/blob/65ba7cbb642a1d56282e7d1b86a728e09a9d6dc5/zerospeech2021/cli/evaluate.py) is the real end of the line in ZS2021, but its functions are not affected by the alterations in `abx_revamped`, so it can be kept as is, separate from the ABX evaluation.

##### Accordingly, the requirements for this match those of the other ABX versions: ABX modules and CPC loading capability. 

## What this is

`abx_revamped` runs on-phone (like phone_ABX), and keeps track of context (like original ABX). Per previous example, the model is given some phone Y and is told that the context was the phones X and Z.

Since we have these two ABX versions' approaches to context, `abx_revamped` offers either "within" context (default original ABX behaviour) or "any" context (default phone_ABX behaviour). These are the two **context modes** - a new variable in this ABX version. 

The code is reworked to accomodate context modes. On the user side, this parallels the way speaker modes were accomodated already, and is integrated with this version of `phonetic.py` (the script that runs the `eval_ABX.py` evaluation for ZS submissions).

## What this will be

At present, `abx_revamped` is a Frankenstein assembly of two sub-versions of the ABX evaluation. Rather than truly streamline the approach to {within, across, any} for any variable, it picks and chooses some combinations, so not all are available. "any" is only available as a context mode, "across" is only available as a speaker mode; therefore we get {within, across} x {within, any} = 4 possible mode combinations.

Right now, the `phone_abx_iterators` code disregards context entirely, hence being called "any" context, while the fundamental `abx_iterators` code only varies for speaker modes and always operates "within" context. 
Neither version has "across context" code, nor does either version have "any speaker" code.

To add these missing options, the ABX evaluation will need a more thorough tearing down.

The {within} context x {across, within} speaker mechanism is written out in the ZS2021 ABX (without specifying context as a variable - just setting context to permanent "within"). The simplest modification would generalise this mechanism to "within X, within/across Y", allowing "within speaker"+"across context".  
Likewise, the {any} context mechanism is written out for {across, within} speaker in phone_ABX, setting context to permanent "any". This too could be generalised to "any X, within/across Y", allowing the "any speaker"+"within/across context" combinations.  
Finally, the "across speaker"+"across context" and "any speaker"+"any context" combinations will need to be written up specially for this `abx_revamped` evaluation.  

The result should be a neat 3x3, handled by different Iterator classes in abx_iterators:  
`{within, across, any}` x `{within, across, any}`  
       
* {within} x {within, across} (ABXWithinWithinGroupIterator, ABXWithinAcrossGroupIterator)  
* {any} x {within, across} (ABXAnyWithinGroupIterator, ABXAnyAcrossGroupIterator)  
* {across} x {across} (ABXAcrossAcrossGroupIterator)  
* {any} x {any} (ABXAnyAnyGroupIterator)

