# ABX_revamped

The ABX phonetic evaluation as used by ZeroSpeech challenge, in the process of being revamped to add "context-type" options (on-triphone, within-context, any-context).

## What this was based on

This repo is a reworking of the ABX phonetic evaluation used by ZS2021 - you can find the bulk of it [here](https://github.com/zerospeech/zerospeech2021/tree/65ba7cbb642a1d56282e7d1b86a728e09a9d6dc5/zerospeech2021). 
The ABX evaluation supports two **speaker modes**: "within" speaker and "across" speaker.

`abx_revamped` mirrors many parts of the `phonetic_eval` directory, which in turn takes after the [libri-light ABX evaluation](https://github.com/facebookresearch/libri-light/tree/main/eval). This version ran on-triphone (the models were given an entire XYZ sequence)
The parts of `abx_revamped` that aren't from ZS2021 or libri-light were borrowed from an on-phone version of ABX (the models were given Y, without X or Z). Hence the name `phone_abx_iterators` and its constituents. 

`abx_revamped` also takes from ZS2021 the file `phonetic.py`, which uses `phonetic_eval`'s contents in ZS-specific ways. [evaluate.py](https://github.com/zerospeech/zerospeech2021/blob/65ba7cbb642a1d56282e7d1b86a728e09a9d6dc5/zerospeech2021/cli/evaluate.py) is the real end of the line in ZS2021, but its functions are not affected by the alterations in `abx_revamped`, so it can be kept as is, separate from the ABX evaluation.

#### Accordingly, the requirements for this match those of the other ABX versions: ABX modules and CPC loading capability. 

## What this is

`abx_revamped` runs on-phone (like phone_ABX), and keeps track of context (like original ABX). Per previous example, the model is given some phone Y and is told that the context was the phones X and Z.

Since we have these two ABX versions' approaches to context, `abx_revamped` offers either "within" context (default original ABX behaviour) or "any" context (default phone_ABX behaviour). These are the two **context modes** - a new variable in this ABX version. 

The code is reworked to accomodate context modes. On the user side, this parallels the way speaker modes were accomodated already, and is integrated with this version of `phonetic.py` (the script that runs the `eval_ABX.py` evaluation for ZS submissions).

## What this will be

At present, `abx_revamped` is a Frankenstein assembly of two sub-versions of the ABX evaluation. Rather than truly streamline the approach to {within, across, any} for any variable, it picks and chooses some combinations, so not all are available. ("any" is only available for contexts, "across" only for speakers, therefore we get 2x2 = 4 total mode combinations).

Right now, the `phone_abx_iterators` code disregards context entirely, hence being called "any" context, while the fundamental `abx_iterators` code only varies for speaker modes and always operates "within" context. 
Neither version has "across context" code, nor does either version have "any speaker" code.

To add these missing options, the ABX evaluation will need a more thorough tearing down.
The [within context] "across" vs "within" speaker mechanism is written out quite well already. The simplest modification would swap "speaker" and "context" at will, so it could be modified for "across context", but would necessarily need to be [within speaker]. It would not cover "across speaker"+"across context".
The "any" speaker option would need to be added into `abx_iterators`, ideally applicable to either the speaker or the context variable.

The result should be a neat 3x3 option: {within, across, any} x {within, across, any}. 


## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.cognitive-ml.fr/alyashenko/abx_revamped.git
git branch -M main
git push -uf origin main
```
