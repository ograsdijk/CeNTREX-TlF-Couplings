from typing import Sequence

import numpy as np
import numpy.typing as npt
from centrex_TlF_hamiltonian import states

from .matrix_elements import generate_ED_ME_mixed_state

__all__ = ["calculate_br"]


def calculate_br(
    excited_state: states.State,
    ground_states: Sequence[states.State],
    tol: float = 1e-2,
) -> npt.NDArray[np.float_]:
    # matrix elements between the excited state and the ground states
    MEs = np.zeros((len(ground_states)), dtype=np.complex_)

    for idg, ground_state in enumerate(ground_states):
        MEs[idg] = generate_ED_ME_mixed_state(
            ground_state.remove_small_components(tol=tol),
            excited_state.remove_small_components(tol=tol),
        )

    # Calculate branching ratios
    BRs = np.abs(MEs) ** 2 / (np.sum(np.abs(MEs) ** 2)).astype(np.float_)
    return BRs
