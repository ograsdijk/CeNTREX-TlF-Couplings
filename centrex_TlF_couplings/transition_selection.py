from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import sympy as smp
from centrex_TlF_hamiltonian import states

__all__ = ["TransitionSelector"]


@dataclass
class TransitionSelector:
    ground: Sequence[states.State]
    excited: Sequence[states.State]
    polarizations: Sequence[npt.NDArray[np.complex_]]
    polarization_symbols: List[smp.Symbol]
    Ω: smp.Symbol
    δ: smp.Symbol
    description: Optional[str] = None
    type: Optional[str] = None
    ground_main: Optional[states.State] = None
    excited_main: Optional[states.State] = None
