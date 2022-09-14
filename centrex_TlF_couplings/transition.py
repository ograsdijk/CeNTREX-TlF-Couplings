from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import sympy as smp
from centrex_TlF_hamiltonian import states
from centrex_TlF_hamiltonian.states import ElectronicState

from centrex_TlF_couplings.utils import check_transition_coupled_allowed

from .polarization import Polarization

__all__ = [
    "TransitionSelector",
    "OpticalTransitionType",
    "MicrowaveTransition",
    "OpticalTransition",
    "generate_transition_selectors",
    "get_possible_optical_transitions",
]


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

    def __repr__(self) -> str:
        if self.description is None:
            J_g = np.unique([g.largest.J for g in self.ground])[0]
            J_e = np.unique([e.largest.J for e in self.excited])[0]
            return f"TransitionSelector(J={J_g} -> J={J_e})"
        else:
            return f"TransitionSelector({self.description})"


class OpticalTransitionType(Enum):
    P = -1
    Q = 0
    R = +1


@dataclass
class MicrowaveTransition:
    J_ground: int
    J_excited: int
    electronic_ground: ElectronicState = ElectronicState.X
    electronic_excited: ElectronicState = ElectronicState.X

    def __repr__(self) -> str:
        return f"MicrowaveTransition({self.name})"

    @property
    def name(self) -> str:
        return f"J={self.J_ground} -> J={self.J_excited}"

    @property
    def Ω_ground(self) -> int:
        return 0

    @property
    def Ω_excited(self) -> int:
        return 0

    @property
    def P_ground(self) -> int:
        return (-1) ** self.J_ground

    @property
    def P_excited(self) -> int:
        return (-1) ** self.J_excited

    @property
    def qn_select_ground(self) -> states.QuantumSelector:
        return states.QuantumSelector(
            J=self.J_ground, electronic=self.electronic_ground, Ω=self.Ω_ground
        )

    @property
    def qn_select_excited(self) -> states.QuantumSelector:
        return states.QuantumSelector(
            J=self.J_excited, electronic=self.electronic_excited, Ω=self.Ω_excited
        )


@dataclass
class OpticalTransition:
    t: OpticalTransitionType
    J_ground: int
    F1: float
    F: int
    electronic_ground: ElectronicState = ElectronicState.X
    electronic_excited: ElectronicState = ElectronicState.B

    def __repr__(self) -> str:
        return f"OpticalTransition({self.name})"

    @property
    def name(self) -> str:
        F1 = smp.S(str(self.F1), rational=True)
        return f"{self.t.name}({self.J_ground}) F1'={F1} F'={self.F}"

    @property
    def J_excited(self) -> int:
        return self.J_ground + self.t.value

    @property
    def P_excited(self) -> int:
        return self.P_ground * -1

    @property
    def P_ground(self) -> int:
        return (-1) ** self.J_ground

    @property
    def Ω_excited(self) -> int:
        return 1

    @property
    def Ω_ground(self) -> int:
        return 0

    @property
    def qn_select_ground(self) -> states.QuantumSelector:
        return states.QuantumSelector(
            J=self.J_ground, electronic=self.electronic_ground, Ω=self.Ω_ground
        )

    @property
    def qn_select_excited(self) -> states.QuantumSelector:
        return states.QuantumSelector(
            J=self.J_excited,
            F1=self.F1,
            F=self.F,
            electronic=self.electronic_excited,
            P=self.P_excited,
            Ω=self.Ω_excited,
        )

    @property
    def ground_states(self) -> Sequence[states.CoupledBasisState]:
        return states.generate_coupled_states_X(self.ground_states)

    @property
    def excited_states(self) -> Sequence[states.CoupledBasisState]:
        return states.generate_coupled_states_B(self.qn_select_excited)


def generate_transition_selectors(
    transitions: Sequence[Sequence[Union[OpticalTransition, MicrowaveTransition]]],
    polarizations: Sequence[Sequence[Polarization]],
    ground_mains: Optional[Sequence[states.State]] = None,
    excited_mains: Optional[Sequence[states.State]] = None,
) -> List[TransitionSelector]:
    """
    Generate a list of TransitionSelectors from Transition(s) and Polarization(s).

    Args:
        transitions (Sequence[Union[OpticalTransition, MicrowaveTransition]]):
                                                    transitions to include in the system
        polarizations (Sequence[Sequence[Polarization]]): polarization, list of
                                                        polarizations per transition.
        ground_mains (Optional[Sequence[States]]): Sequence of a main ground state to
                                                    use per transition
        excited_mains (Optional[Sequence[States]]): Sequence of a a main excited state
                                                    to use per transition
        excited_mains:

    Returns:
        List[TransitionSelector]: List of TransitionSelectors
    """
    transition_selectors = []

    for idt, (transition, polarization) in enumerate(zip(transitions, polarizations)):
        ground_states_approx_qn_select = states.QuantumSelector(
            J=transition.J_ground,
            electronic=transition.electronic_ground,
            P=transition.P_ground,
            Ω=transition.Ω_ground,
        )
        ground_states_approx = list(
            1 * states.generate_coupled_states_X(ground_states_approx_qn_select)
        )

        if isinstance(transition, OpticalTransition):
            excited_states_approx = list(
                1 * states.generate_coupled_states_B(transition.qn_select_excited)
            )
        elif isinstance(transition, MicrowaveTransition):
            excited_states_approx = list(
                1 * states.generate_coupled_states_X(transition.qn_select_excited)
            )

        transition_selectors.append(
            TransitionSelector(
                ground=ground_states_approx,
                excited=excited_states_approx,
                polarizations=[p.vector for p in polarization],
                polarization_symbols=[
                    smp.Symbol(f"P{p.name}{idt}") for p in polarization
                ],
                Ω=smp.Symbol(f"Ω{idt}", complex=True),
                δ=smp.Symbol(f"δ{idt}"),
                description=transition.name,
                ground_main=None if ground_mains is None else ground_mains[idt],
                excited_main=None if excited_mains is None else excited_mains[idt],
            )
        )
    return transition_selectors


def get_possible_optical_transitions(
    ground_state: states.CoupledBasisState,
    transition_types: Optional[Sequence[OpticalTransitionType]] = None,
):
    J = ground_state.J
    F1 = ground_state.F1
    F = ground_state.F
    I1 = ground_state.I1
    I2 = ground_state.I2

    if transition_types is None:
        transition_types = [t for t in OpticalTransitionType]

    transitions = []
    for transition_type in transition_types:
        ΔJ = transition_type.value
        J_excited = J + ΔJ
        _transitions = [
            OpticalTransition(transition_type, J, F1, F)
            for F1 in np.arange(np.abs(J_excited - I1), J_excited + I1 + 1)
            for F in np.arange(np.abs(F1 - I2), F1 + I2 + 1, dtype=int)
        ]
        _transitions = [
            t
            for t in _transitions
            if check_transition_coupled_allowed(ground_state, t.excited_states[0])
        ]
        transitions.append(_transitions)
    return transitions
