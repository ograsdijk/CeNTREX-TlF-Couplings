from typing import List, Sequence, Tuple, Union
from centrex_TlF_hamiltonian.states.states import CoupledBasisState

import numpy as np
import numpy.typing as npt
from centrex_TlF_hamiltonian import states

__all__: List[str] = []


def check_transition_coupled_allowed(
    state1: states.CoupledBasisState,
    state2: states.CoupledBasisState,
    ΔmF_allowed: int,
    return_err: float = True,
) -> Union[bool, Tuple[bool, str]]:
    """Check whether the transition is allowed based on the quantum numbers

    Args:
        state1 (CoupledBasisState): ground CoupledBasisState
        state2 (CoupledBasisState): excited CoupledBasisState
        ΔmF_allowed (int): allowed ΔmF for the transition
        return_err (boolean): boolean flag for returning the error message

    Returns:
        tuple: (allowed boolean, error message)
    """
    assert state1.P is not None, "parity is required to be set for state1"
    assert state2.P is not None, "parity is required to be set for state2"

    ΔF = int(state2.F - state1.F)
    ΔmF = np.abs(int(state2.mF - state1.mF))
    ΔP = int(state2.P - state1.P)

    flag_ΔP = abs(ΔP) != 2
    flag_ΔF = abs(ΔF) > 1
    flag_ΔmF = ΔmF != ΔmF_allowed
    flag_ΔFΔmF = (not flag_ΔmF) & ((ΔF == 0) & (ΔmF == 0) & (state1.mF == 0))

    errors = ""
    if flag_ΔP:
        errors += "parity invalid"
    if flag_ΔF:
        if len(errors) != 0:
            errors += ", "
        errors += f"ΔF invalid -> ΔF = {ΔF}"
    if flag_ΔmF:
        if len(errors) != 0:
            errors += ", "
        errors += f"ΔmF invalid -> ΔmF = {ΔmF}"

    if flag_ΔFΔmF:
        if len(errors) != 0:
            errors += ", "
        errors += "ΔF = 0 & ΔmF = 0 invalid"

    if len(errors) != 0:
        errors = f"transition not allowed; {errors}"

    if return_err:
        return not (flag_ΔP | flag_ΔF | flag_ΔmF | flag_ΔFΔmF), errors
    else:
        return not (flag_ΔP | flag_ΔF | flag_ΔmF | flag_ΔFΔmF)


def assert_transition_coupled_allowed(
    state1: states.CoupledBasisState, state2: states.CoupledBasisState, ΔmF_allowed: int
) -> bool:
    """Check whether the transition is allowed based on the quantum numbers.
    Raises an AssertionError if the transition is not allowed.

    Args:
        state1 (CoupledBasisState): ground CoupledBasisState
        state2 (CoupledBasisState): excited CoupledBasisState

    Returns:
        tuple: allowed boolean
    """
    ret = check_transition_coupled_allowed(state1, state2, ΔmF_allowed, return_err=True)
    if isinstance(ret, tuple):
        allowed, errors = ret
        assert allowed, errors
        return allowed
    else:
        return ret


def select_main_states(
    ground_states: Sequence[states.State],
    excited_states: Sequence[states.State],
    polarization: npt.NDArray[np.complex_],
) -> Tuple[states.State, states.State]:
    """Select main states for calculating the transition strength to normalize
    the Rabi rate with

    Args:
        ground_states (Sequence[states.State]): Sequence of ground states for the
                                                transition
        excited_states (Sequence[states.State]): Sequence of excited states for the
                                                transition
        polarization (npt.NDArray[np.float_]): polarization vector
    """
    ΔmF = 0 if polarization[2] != 0 else 1

    allowed_transitions = []
    for ide, exc in enumerate(excited_states):
        exc_basisstate: CoupledBasisState = exc.largest  # type: ignore
        for idg, gnd in enumerate(ground_states):
            gnd_basisstate: CoupledBasisState = gnd.largest  # type: ignore
            if check_transition_coupled_allowed(
                gnd_basisstate, exc_basisstate, ΔmF, return_err=False
            ):
                allowed_transitions.append((idg, ide, exc_basisstate.mF))

    assert (
        len(allowed_transitions) > 0
    ), "none of the supplied ground and excited states have allowed transitions"

    excited_state = excited_states[allowed_transitions[0][1]]
    ground_state = ground_states[allowed_transitions[0][0]]

    return ground_state, excited_state
