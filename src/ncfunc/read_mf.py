from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import core


def read_mf_within(
    paths: Sequence[core.PathLike],
    var_name: str,
    withins: tuple[tuple[float | None, float | None], ...],
    idim_time: int | None = None,
    decode_time: bool = True,
    stacked_along: None | int = None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """
    read a variable within bounds from multiple files and combine the results

    each file is read with core.read_within() under the same withins, then the
    per-file values are joined. with stacked_along=None a new leading axis of
    length len(paths) is prepended (one slice per file, in path order); with an
    integer stacked_along the values are concatenated along that dimension and
    the matching coordinate array is concatenated too.

    the remaining dimensions' coordinates are taken from the last file; the
    combined data is expected to align, so callers should ensure the
    non-stacked coordinates are identical across files.

    returns (values, coords) with coords the same shape as core.read_within()
    but for the stacked dimension replaced or extended by concatenation.

    raises RuntimeError when fewer than 2 paths are given, ValueError /
    KeyError / FileNotFoundError as reported by core.read_within()
    """
    if len(paths) == 0:
        raise RuntimeError("zero path is given.")
    elif len(paths) == 1:
        raise RuntimeError("Only 1 path is given. Use `read_within()` instead.")

    paths = tuple(Path(p) for p in paths)

    values = []
    stacked_dim = []

    for path in paths:
        value, dims = core.read_within(path, var_name, withins, idim_time, decode_time)
        values.append(value)

        if stacked_along is not None:
            stacked_dim.append(dims[stacked_along])

    if stacked_along is None:
        value_out = np.stack(values, axis=0)
        dims_out = (np.arange(len(paths)), *dims)
    else:
        value_out = np.concatenate(values, axis=stacked_along)
        dims_out = list(dims)
        dims_out[stacked_along] = np.concatenate(stacked_dim)
        dims_out = tuple(dims_out)

    return value_out, dims_out
