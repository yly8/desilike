from .des import DESWeakLensing3x2pt
from .des_k import DESKDependentWeakLensing3x2pt


def __getattr__(name):
    if name == "DESY3Theory":
        from .des_y3 import DESY3Theory
        return DESY3Theory
    raise AttributeError(name)
