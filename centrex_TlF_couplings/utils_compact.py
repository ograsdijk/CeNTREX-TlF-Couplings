from typing import List

import numpy as np
import numpy.typing as npt

__all__: List[str] = []


def delete_row_column_indices(
    arr: npt.NDArray[np.float_], indices_compact: npt.NDArray[np.int_]
) -> npt.NDArray[np.float_]:
    return np.delete(
        np.delete(arr, indices_compact[1:], axis=0), indices_compact[1:], axis=1
    )


def BR_to_C_array(BR: npt.NDArray[np.float_], Gamma) -> npt.NDArray[np.float_]:
    return np.sqrt(BR * Gamma)


def C_array_to_BR(
    C_array: npt.NDArray[np.float_], Gamma: float
) -> npt.NDArray[np.float_]:
    return C_array ** 2 / Gamma


def compact_BR_array_indices(
    BR_array: npt.NDArray[np.float_], indices_compact: npt.NDArray[np.int_]
) -> npt.NDArray[np.float_]:
    new_shape = np.asarray(BR_array[0].shape) - len(indices_compact) + 1
    BR_array_new = []
    for BR in BR_array:
        id1, id2 = np.nonzero(BR)
        id1 = id1[0]
        id2 = id2[0]
        if (id1 not in indices_compact) & (id2 not in indices_compact):
            BR_array_new.append(delete_row_column_indices(BR, indices_compact))

    BR_sum = np.zeros(new_shape, "complex")
    BR_sum[indices_compact[0], :] = np.sum(BR_array, axis=0)[indices_compact].sum(
        axis=0
    )[-new_shape[0] :]
    for id1, id2 in zip(*np.nonzero(BR_sum)):
        BR_new = np.zeros(new_shape, "complex")
        BR_new[id1, id2] = BR_sum[id1, id2]
        BR_array_new.append(BR_new)
    return np.array(BR_array_new)


def compact_C_array_indices(
    C_array: npt.NDArray[np.float_], Gamma: float, indices_compact: npt.NDArray[np.int_]
) -> npt.NDArray[np.float_]:
    BR = C_array_to_BR(C_array, Gamma)
    BR_compact = compact_BR_array_indices(BR, indices_compact)
    C_array_compact = BR_to_C_array(BR_compact, Gamma)
    return C_array_compact
