from pathlib import Path

import datenum as dn
import numpy as np
import pytest

import ncfunc as ncf
from ncfunc import read_mf


def _save_tile(path: Path, i: int, ntime: int = 2, ny: int = 3, nx: int = 2) -> None:
    """write one time-split file; var(t,y,x) = 100*i + 1000*t + 10*y + x."""
    times = list(range(i * ntime, (i + 1) * ntime))
    var = [
        [[100 * i + 1000 * t + 10 * y + x for x in range(nx)] for y in range(ny)]
        for t in range(ntime)
    ]
    ncf.save(
        path, {"var": var, "time": times, "y": list(range(ny)), "x": list(range(nx))}
    )


def _save_spatial_tile(path: Path, i: int, ny: int = 3, nx: int = 2) -> None:
    """write one x-tiled file with no time dim; var(y,x) = 100*i + 10*y + x."""
    xs = list(range(i * nx, (i + 1) * nx))
    var = [[100 * i + 10 * y + x for x in xs] for y in range(ny)]
    ncf.save(path, {"var": var, "y": list(range(ny)), "x": xs})


@pytest.fixture
def tiles(tmp_path: Path) -> list[Path]:
    paths = []
    for i in range(3):
        path = tmp_path / f"tile{i}.nc"
        _save_tile(path, i)
        paths.append(path)
    return paths


@pytest.fixture
def spatial_tiles(tmp_path: Path) -> list[Path]:
    paths = []
    for i in range(3):
        path = tmp_path / f"spatial{i}.nc"
        _save_spatial_tile(path, i)
        paths.append(path)
    return paths


TIME_ALL = ((None, None), (None, None), (None, None))
SPATIAL_ALL = ((None, None), (None, None))


def test_zero_paths_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="zero path"):
        read_mf.read_mf_within([], "var", TIME_ALL)


def test_single_path_raises(tmp_path: Path) -> None:
    path = tmp_path / "single.nc"
    _save_tile(path, 0)
    with pytest.raises(RuntimeError, match="Only 1 path"):
        read_mf.read_mf_within([path], "var", TIME_ALL)


def test_stacked_along_new_axis(tiles: list[Path]) -> None:
    value, dims = read_mf.read_mf_within(tiles, "var", TIME_ALL, stacked_along=None)

    assert value.shape == (3, 2, 3, 2)  # one leading entry per file
    assert np.array_equal(dims[0], [0, 1, 2])  # one entry per file

    # each leading item is the raw, unsliced data of that file
    assert np.array_equal(value[0][0], np.array([[0, 1], [10, 11], [20, 21]]))
    assert np.array_equal(value[1][0], np.array([[100, 101], [110, 111], [120, 121]]))


def test_stacked_along_time_axis(tiles: list[Path]) -> None:
    value, dims = read_mf.read_mf_within(tiles, "var", TIME_ALL, stacked_along=0)

    assert value.shape == (6, 3, 2)  # concatenated along time
    assert len(dims[0]) == 6

    # times decode to datenums: days 0..5 since 2000-01-01
    assert dims[0][0] == dn.from_ymd(2000, 1, 1)
    assert dims[0][-1] == dn.from_ymd(2000, 1, 6)

    # data is the concatenation of all files, slice order preserved
    assert np.array_equal(value[0], np.array([[0, 1], [10, 11], [20, 21]]))
    assert np.array_equal(value[2], np.array([[100, 101], [110, 111], [120, 121]]))
    assert np.array_equal(value[4], np.array([[200, 201], [210, 211], [220, 221]]))


def test_within_range_on_time_axis(tiles: list[Path]) -> None:
    # keep days 2000-01-02..05, which slice a couple of days out of every file
    value, dims = read_mf.read_mf_within(
        tiles,
        "var",
        (
            (dn.from_ymd(2000, 1, 2), dn.from_ymd(2000, 1, 5)),
            (None, None),
            (None, None),
        ),
        stacked_along=0,
    )

    assert value.shape == (4, 3, 2)  # four days total across the files
    assert dims[0][0] == dn.from_ymd(2000, 1, 2)
    assert dims[0][-1] == dn.from_ymd(2000, 1, 5)


def test_stacked_along_space_axis(spatial_tiles: list[Path]) -> None:
    value, dims = read_mf.read_mf_within(
        spatial_tiles, "var", SPATIAL_ALL, stacked_along=1, decode_time=False
    )

    assert value.shape == (3, 6)  # 3 y-rows, concatenated x columns
    assert np.array_equal(dims[0], [0, 1, 2])  # the y coordinate
    assert np.array_equal(dims[1], [0, 1, 2, 3, 4, 5])  # concatenated x

    # the concatenated data forms a contiguous x grid across the tiles
    assert np.array_equal(value[0], [0, 1, 102, 103, 204, 205])


def test_within_range_on_space_axis(spatial_tiles: list[Path]) -> None:
    # keep only x in [1, 4] from each file before stacking along x
    value, dims = read_mf.read_mf_within(
        spatial_tiles,
        "var",
        ((None, None), (1, 4)),
        stacked_along=1,
        decode_time=False,
    )

    # per-file x slices: tile0 keeps {1}, tile1 keeps {2,3}, tile2 keeps {4}
    assert value.shape == (3, 4)
    assert np.array_equal(dims[1], [1, 2, 3, 4])
    assert np.array_equal(value[0], [1, 102, 103, 204])
