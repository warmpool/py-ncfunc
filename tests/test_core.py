from pathlib import Path
from typing import Any

import datenum as dn
import numpy as np
import pytest

import ncfunc as ncf
from ncfunc import core
from tests import nctools as nct

TEST_FILE = "tests/ersst_2022-2024.nc"
TEST_FILE_LINK = "../ncplot/tests/data/file1.nc"


# ---- helpers ---
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


# ---- meta data queries ----
def test_var_names() -> None:
    assert ncf.var_names(TEST_FILE) == ("time", "lon", "lat", "sst", "ssta")


def test_dim_names() -> None:
    assert ncf.dim_names(TEST_FILE, "ssta") == ("time", "lat", "lon")
    assert ncf.dim_names(TEST_FILE, "sst") == ("time", "lat", "lon")
    assert ncf.dim_names(TEST_FILE, "lon") == ("lon",)

    with pytest.raises(KeyError):
        ncf.dim_names(TEST_FILE, "no-such-variable")


def test_shape() -> None:
    assert ncf.shape(TEST_FILE, "lon") == (240,)
    assert ncf.shape(TEST_FILE, "lat") == (121,)
    assert ncf.shape(TEST_FILE, "time") == (36,)
    assert ncf.shape(TEST_FILE, "sst") == (36, 121, 240)
    assert ncf.shape(TEST_FILE, "ssta") == (36, 121, 240)


def test_ndim() -> None:
    assert ncf.ndim(TEST_FILE, "lon") == 1
    assert ncf.ndim(TEST_FILE, "lat") == 1
    assert ncf.ndim(TEST_FILE, "time") == 1
    assert ncf.ndim(TEST_FILE, "sst") == 3
    assert ncf.ndim(TEST_FILE, "ssta") == 3


def test_attr_names() -> None:
    assert ncf.attr_names(TEST_FILE, "sst") == (
        "standard_name",
        "long_name",
        "units",
        "_FillValue",
        "missing_value",
        "cell_methods",
    )

    root_attrs = ncf.attr_names(TEST_FILE, "/")

    for some_attr in ("creator_name", "creator_email", "creator_type"):
        assert some_attr in root_attrs


def test_attr_val() -> None:
    def sub_test(path: str, var_name: str, gold: dict[str, Any]) -> None:
        for attr_name, attr_val in gold.items():
            assert ncf.attr_val(path, var_name, attr_name) == attr_val

    sub_test(
        TEST_FILE,
        "time",
        gold={
            "standard_name": "time",
            "long_name": "Time",
            "units": "days since 1850-01-15 00:00",
            "calendar": "gregorian",
            "axis": "T",
        },
    )

    sub_test(
        TEST_FILE,
        "/",
        gold={
            "creator_name": "Boyin Huang",
            "creator_email": "boyin.huang@noaa.gov",
            "creator_type": "group",
        },
    )


def test_var_name_includes() -> None:
    assert ncf.core.var_names_include(TEST_FILE, ("time", "date"), (1,)) == ("time",)

    assert ncf.core.var_names_include(TEST_FILE, ("sst",), (3,), (1, 2)) == (
        "sst",
        "ssta",
    )

    with pytest.raises(ValueError):
        ncf.core.var_names_include(TEST_FILE, ("sst",), (1,))
        ncf.core.var_names_include(TEST_FILE, ("sst",), (3,))


def test_attr_name_includes() -> None:
    assert ncf.core.attr_names_include(TEST_FILE, "time", ("unit"), (1,)) == ("units",)
    assert ncf.core.attr_names_include(TEST_FILE, "time", ("name"), (2,)) == (
        "standard_name",
        "long_name",
    )


def test_read() -> None:
    with pytest.raises(ValueError):
        ncf.read(TEST_FILE, "sst", (slice(0), slice(0)))

    with pytest.raises(ValueError):
        ncf.read(TEST_FILE, "lon", (slice(0), slice(0)))

    for var_name in ncf.var_names(TEST_FILE):
        assert np.array_equal(
            ncf.read(TEST_FILE, var_name), nct.read(TEST_FILE, var_name)
        )

    slices = (slice(5, 10), slice(10, 30), slice(-5))
    assert np.array_equal(
        ncf.read(TEST_FILE, "sst", slices),
        nct.ncread(TEST_FILE, "sst", list(slices)),
    )


def test_read_time() -> None:
    time = ncf.read_time(TEST_FILE)
    ymd = dn.to_ymd(time)[:3]
    assert ymd[0][0] == 2022
    assert ymd[0][-1] == 2024
    assert np.array_equal(ncf.read_time(TEST_FILE), nct.ncreadtime(TEST_FILE))


def test_read_within() -> None:
    sample, dims = ncf.read_within(TEST_FILE, "lon", ((120, 150),))
    assert sample.shape == (21,)
    assert len(dims) == 1
    assert dims[0].shape == (21,)

    sample, dims = ncf.read_within(TEST_FILE, "time", ((None, None),))
    assert sample.shape == (36,)
    sample, dims = ncf.read_within(
        TEST_FILE, "time", ((dn.from_ymd(2024, 1, 15), dn.from_ymd(2024, 12, 16)),)
    )
    assert dims[0][0] == dn.from_ymd(2024, 1, 15)
    assert dims[0][-1] == dn.from_ymd(2024, 12, 15)
