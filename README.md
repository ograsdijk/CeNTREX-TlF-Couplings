# CeNTREX-TlF-Couplings
Code for generating the CeNTREX TlF couplings. 
Includes code for generating branching ratios, electric dipole coupling elements and coupling fields

## Dependencies
* `numpy`
* `centrex_tlf_hamiltonian`

## Installation
`python -m pip install .`  
where `.` is the path to the directory. To install directly from `Github` use:  
`python -m pip install git+https://github.com/ograsdijk/CeNTREX-TlF-Couplings`

## Generating branching ratios
The code below generates branching ratios from `|J'=1, F1'=1/2, mF=0>` to all states in the `J=1` manifold.
```Python
from centrex_tlf_hamiltonian import states
import centrex_tlf_couplings as couplings

excited_state = states.CoupledBasisState(
    J=1, F1=1 / 2, F=1, mF=0, I1=1 / 2, I2=1 / 2, Omega=1, P=1
)
qn_select = states.QuantumSelector(J=1)
ground_states = states.generate_coupled_states_X(qn_select)
br = couplings.calculate_br(1 * excited_state, 1 * ground_states)
```

## Generating couplings
The code below generates the coupling fields for the `J=1` manifold to the `J'=1, F1'=1/2, F'=1` manifold. The returned value is a dataclass `CouplingFields` containing the following fields:  
* `ground_main`
* `excited_main`
* `main_coupling`: the electric dipole coupling between `ground_main` and `excited_main`
* `ground_states`: list of all ground states
* `excited_states`: list of all excited states
* `fields`: a list of `CouplingField` dataclasses with the following fields:  
  * `polarization`: polarization vector
  * `field`: coupling field in the `ground_states` + `excited_states` basis

```Python
from centrex_tlf_hamiltonian import states
import centrex_tlf_couplings as couplings

qn_select = states.QuantumSelector(J=1)
ground_states = states.generate_coupled_states_X(qn_select)

qn_select = states.QuantumSelector(J=1, F1=1 / 2, F=1, P=1, Ω=1)
excited_states = states.generate_coupled_states_B(qn_select)

# the generate_coupling_field_* functions requires lists as inputs, not np.ndarrays
QN = list(1 * np.append(ground_states, excited_states))
ground_states = list(1 * ground_states)
excited_states = list(1 * excited_states)

H_rot = np.eye(len(QN), dtype=complex) * np.arange(len(QN))
V_ref = np.eye(len(QN))
pol_vecs = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])]
normalize_pol = True

coupling = couplings.generate_coupling_field_automatic(
    ground_states, excited_states, H_rot, QN, V_ref, pol_vecs, normalize_pol
)
```
