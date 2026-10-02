"""Location of the bundled DES data, independent of the checkout and working directory."""
from pathlib import Path


def resolve_data_dir(data_dir=None):
    """Return an absolute data directory; an explicit path overrides bundled data.

    Relative custom paths are relative to the working directory and ``~`` is expanded.
    The default is resolved from this installed module, not from a machine-specific path.
    """
    path = Path(__file__).resolve().parent / 'des3' if data_dir is None else Path(data_dir).expanduser()
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'DES data directory does not exist: {path}')
    return path
