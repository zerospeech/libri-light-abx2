from importlib.metadata import version, PackageNotFoundError
from zrc_abx2.eval_ABX import EvalArgs, EvalABX

try:
    __version__ = version("zerospeech-libriabx2")
except PackageNotFoundError:
    # package is not installed
    __version__ = None
