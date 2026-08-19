from pathlib import Path
from typing import Any

import datenum as dn
import netCDF4 as nc
import numpy as np
import pytest

import ncfunc as ncf
from ncfunc import core
from tests import nctools as nct

TEST_FILE = "tests/ersst_2022-2024.nc"
TEST_FILE_LINK = "../ncplot/tests/data/file1.nc"


@pytest.fixture
def tmp_nc(tmp_path: Path) -> Path:
    """Create a temporary NetCDF file with sample data."""
    path = tmp_path / "test.nc"
    with nc.Dataset(path, "w") as h:
        h.createDimension("x", 5)
        h.createDimension("y", 3)
        h.createDimension("t", 4)
        var = h.createVariable("data", np.float32, ("t", "y", "x"))
        var[:] = np.random.rand(4, 3, 5)
        h.createVariable("lon", np.float64, ("x",))
        h["lon"][:] = np.arange(5)
        h.createVariable("lat", np.float64, ("y",))
        h["lat"][:] = np.arange(3)
        h.createVariable("time", np.float64, ("t",))
        h["time"][:] = np.arange(4)
        h["time"].units = "days since 2020-01-01"
        h["time"].calendar = "standard"
        h.setncattr("title", "test file")
    return path


# ---- helpers ---
def test_abort_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        core.DatasetMeta("no-such-path")


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


# ---- write tests ----
def test_write_attr(tmp_nc: Path) -> None:
    core.write_attr(tmp_nc, "data", "long_name", "test data")
    assert ncf.attr_val(tmp_nc, "data", "long_name") == "test data"

    with nc.Dataset(tmp_nc, "a") as h:
        h.setncattr("author", "tester")
    assert ncf.attr_val(tmp_nc, "/", "author") == "tester"

    with pytest.raises((KeyError, FileNotFoundError)):
        core.write_attr(tmp_nc, "nonexistent", "attr", "val")


def test_write(tmp_nc: Path) -> None:
    original = ncf.read(tmp_nc, "data")
    core.write(tmp_nc, "data", original)
    assert np.array_equal(ncf.read(tmp_nc, "data"), original)

    subset_data = original[:2, :3, :5]
    core.write(tmp_nc, "data", subset_data, (slice(0, 2), slice(0, 3), slice(0, 5)))
    assert np.array_equal(ncf.read(tmp_nc, "data", (slice(0, 2), slice(0, 3), slice(0, 5))), subset_data)

    with pytest.raises(ValueError):
        core.write(tmp_nc, "data", np.array([]))

    with pytest.raises(ValueError):
        core.write(tmp_nc, "data", np.zeros((2, 2)))

    with pytest.raises(ValueError):
        core.write(tmp_nc, "data", np.zeros((2, 3, 5)), (slice(0),))


def test_create(tmp_nc: Path) -> None:
    core.create(tmp_nc, "data", {"t": 4, "y": 3, "x": 5})
    assert "data" in ncf.var_names(tmp_nc)

    new_path = tmp_nc.parent / "new.nc"
    with nc.Dataset(new_path, "w") as h:
        h.createDimension("time", 10)
    core.create(new_path, "time", {"time": 10})
    assert "time" in ncf.var_names(new_path)

    with pytest.raises(ValueError):
        core.create(tmp_nc, "data", {"t": 99, "y": 99, "x": 99})


def test_set_my_attrs(tmp_nc: Path) -> None:
    with nc.Dataset(tmp_nc, "a") as h:
        core._set_my_attrs(h, "lon")
        assert h["lon"].axis == "X"
        assert h["lon"].units == "degrees_east"

        core._set_my_attrs(h, "lat")
        assert h["lat"].axis == "Y"
        assert h["lat"].units == "degrees_north"

        core._set_my_attrs(h, "time")
        assert h["time"].axis == "T"
        assert h["time"].units == "days since 2000-01-01 00:00:00"


def test_read_nonexistent_var(tmp_nc: Path) -> None:
    with pytest.raises(KeyError):
        ncf.read(tmp_nc, "nonexistent")


def test_read_within_mismatched_ndim(tmp_nc: Path) -> None:
    with pytest.raises(ValueError):
        ncf.read_within(tmp_nc, "data", ((1, 2),))


def test_read_within_idim_time_with_decode_false(tmp_nc: Path) -> None:
    with pytest.raises(ValueError):
        ncf.read_within(tmp_nc, "data", ((None, None),) * 3, idim_time=0, decode_time=False)


def test_read_within_idim_time_out_of_range(tmp_nc: Path) -> None:
    with pytest.raises(IndexError):
        ncf.read_within(tmp_nc, "data", ((None, None),) * 3, idim_time=99)


def test_attr_val_nonexistent(tmp_nc: Path) -> None:
    with pytest.raises(KeyError):
        ncf.attr_val(tmp_nc, "data", "nonexistent_attr")
