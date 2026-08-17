from pathlib import Path

import numpy as np
import pytest

import ncfunc as ncf
from ncfunc import core

TEST_FILE = "tests/ersst_2022-2024.nc"
TEST_FILE_LINK = "../ncplot/tests/data/file1.nc"


def test_abort_missing_file() -> None:
    core._abort_missing_path(Path(TEST_FILE))
    with pytest.raises(FileNotFoundError):
        core._abort_missing_path(Path("no-such-path"))


def test_meta_data() -> None:
    ds = core.DatasetMeta(TEST_FILE)
    assert tuple(v for v in ds.variables) == ("time", "lon", "lat", "sst", "ssta")

    assert ds.mtime == 1785817424.361744

    assert ds.attributes["creator_name"] == "Boyin Huang"
    assert ds.attributes["creator_email"] == "boyin.huang@noaa.gov"
    assert ds.attributes["creator_type"] == "group"

    assert ds.variables["time"].dimensions == ("time",)
    assert ds.variables["lat"].dimensions == ("lat",)
    assert ds.variables["lon"].dimensions == ("lon",)
    assert ds.variables["sst"].dimensions == ("time", "lat", "lon")
    assert ds.variables["ssta"].dimensions == ("time", "lat", "lon")

    assert ds.variables["lon"].shape == (240,)
    assert ds.variables["sst"].shape == (36, 121, 240)

    assert ds.variables["sst"].attributes == {
        "standard_name": "sea_surface_temperature",
        "long_name": "Extended reconstructed sea surface temperature",
        "units": "degree_C",
        "_FillValue": np.float32(-999.0),
        "missing_value": np.float32(-999.0),
        "cell_methods": "lev: mean",
    }
    assert ds.variables["time"].attributes == {
        "standard_name": "time",
        "long_name": "Time",
        "units": "days since 1850-01-15 00:00",
        "calendar": "gregorian",
        "axis": "T",
    }


def test_cache() -> None:
    ds1 = core.DatasetMeta.try_from_cache(TEST_FILE)
    assert len(core.DatasetMeta.cached_registry) == 1
    ds2 = core.DatasetMeta.try_from_cache(TEST_FILE, cache_only=True)
    assert len(core.DatasetMeta.cached_registry) == 1
    assert ds1 == ds2
    core.DatasetMeta.try_from_cache(TEST_FILE)
    assert len(core.DatasetMeta.cached_registry) == 1
    core.DatasetMeta.try_from_cache(TEST_FILE_LINK)
    assert len(core.DatasetMeta.cached_registry) == 1


def test_var_names() -> None:
    assert ncf.var_names(TEST_FILE) == ("time", "lon", "lat", "sst", "ssta")


def test_dim_names() -> None:
    assert ncf.dim_names(TEST_FILE, "ssta") == ("time", "lat", "lon")
    assert ncf.dim_names(TEST_FILE, "sst") == ("time", "lat", "lon")
    assert ncf.dim_names(TEST_FILE, "lon") == ("lon",)

    with pytest.raises(KeyError):
        ncf.dim_names(TEST_FILE, "no-such-variable")
