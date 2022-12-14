"""Evaluation program for ZR2021 submissions
(Modified for only phonetic evaluation, and in the process of
    being changed around to accept CPC checkpoints in the future)"""
import atexit
import os
import pathlib
import shutil
import sys
import tempfile
import zipfile

import click
import pandas
import yaml

import phonetic


def write_csv(frame, filename):
    frame.to_csv(filename, index=False, float_format='%.4f')
    print(f'  > Wrote {filename}')


def eval_phonetic(dataset, submission, output, kinds, force_cpu):
    try:
        meta = yaml.safe_load((submission / 'meta.yaml').open('r').read())
        metric = meta['parameters']['phonetic']['metric']
        frame_shift = meta['parameters']['phonetic']['frame_shift']
    except: 
        # resorting to defaults for CPC checkpoints
        metric = cosine
        frame_shift = 100

    results = []
    for kind in kinds:  # 'dev' or 'test'
        results.append(phonetic.evaluate(
            submission / 'phonetic', dataset / 'phonetic',
            kind, metric, frame_shift, force_cpu=force_cpu))

    write_csv(pandas.concat(results), output / 'score_phonetic.csv')


@click.command(epilog='See https://zerospeech.com/2021 for more details')
@click.argument('dataset', type=pathlib.Path)
@click.argument('submission', type=pathlib.Path)
@click.option(
    '-j', '--njobs', default=1, type=int,
    help='Parallel jobs to use for semantic part (default to 1)')
@click.option(
    '--force-cpu', help='Do not use GPU for phonetic part', is_flag=True)
@click.option(
    '-o', '--output-directory', type=pathlib.Path,
    default='.', show_default=True,
    help="Directory to store output results")
def evaluate(
        dataset, submission, njobs, force_cpu, output_directory,
        no_phonetic, no_lexical, no_syntactic, no_semantic):
    """Evaluate a submission to the Zero Resource Speech Challenge 2021
    DATASET is the root directory of the ZR2021 dataset, as downloaded from
    https://zerospeech.com/2021.
    SUBMISSION is the submission to evaluate, it can be a .zip file or a
    directory.
    """
    try:
        kinds = ['dev', 'test']

        # ensures the dataset exists
        dataset = dataset.resolve(strict=True)
        if not dataset.is_dir():
            raise ValueError(f'dataset not found: {dataset}')

        # ensures the submission exists, it it is a zip, uncompress it
        submission = submission.resolve(strict=True)
        if submission.is_file() and zipfile.is_zipfile(submission):
            # create a temp directory we remove at exit
            submission_unzip = tempfile.mkdtemp()
            atexit.register(shutil.rmtree, submission_unzip)

            # uncompress to the temp directory
            print(f'Unzip submission to {submission_unzip}...')
            zipfile.ZipFile(submission, 'r').extractall(submission_unzip)
            submission = pathlib.Path(submission_unzip)
        elif not submission.is_dir():
            raise ValueError(
                f'submssion is not a zip file or a directory: {submission}')

        if not output_directory.is_dir():
            output_directory.mkdir(exist_ok=True, parents=True)

    except ValueError as error:
        print(f'ERROR: {error}')
        sys.exit(-1)
