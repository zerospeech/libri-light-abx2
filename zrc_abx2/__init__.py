from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("zerospeech-libriabx2")
except PackageNotFoundError:
    # package is not installed
    __version__ = None
